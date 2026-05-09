# Copyright (c) OpenMMLab. All rights reserved.
"""Reliability-guided cross/self attention residual fusion modules."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ReliabilityGuidedAttentionResidualFusion(nn.Module):
    """Lightweight residual correction after the original BEVFusion fuser.

    This module does not replace the original ConvFuser output. It predicts a
    residual correction from camera/LiDAR BEV features and adds it to the
    already-fused feature:

        F_final = F_base + gate * A

    Args:
        in_channels: ``[camera_channels, lidar_channels]``.
        out_channels: Channel count of the fused BEV feature.
        use_attention_fusion: If false, return ``F_base`` unchanged.
        use_self_attention: Add BEV self-attention after cross-attention.
        use_reliability_gate: Use a degradation gate.
        gate_type: ``scalar`` uses one learnable gate. ``feature`` predicts a
            sample-wise gate from camera/LiDAR BEV feature statistics.
            ``proxy`` uses explicit camera/LiDAR reliability proxies computed
            from the current sensor inputs.
        hidden_dim: Attention hidden channel count.
        num_heads: Number of attention heads.
        downsample_ratio: Spatial downsampling ratio before attention.
        gate_init: Initial scalar before sigmoid. ``-6`` is nearly off.
        debug_shape: Print tensor shapes once per process.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 use_attention_fusion: bool = False,
                 use_self_attention: bool = True,
                 use_reliability_gate: bool = True,
                 gate_type: str = 'scalar',
                 gate_hidden_dim: int = 64,
                 proxy_gate_min: float = 0.0,
                 proxy_gate_scale: float = 0.1,
                 proxy_gate_power: float = 1.0,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 downsample_ratio: int = 4,
                 residual_loss_weight: float = 0.0,
                 gate_loss_weight: float = 0.0,
                 gate_init: float = -6.0,
                 debug_shape: bool = False) -> None:
        super().__init__()
        assert len(in_channels) == 2, in_channels
        assert hidden_dim % num_heads == 0
        self.use_attention_fusion = use_attention_fusion
        self.use_self_attention = use_self_attention
        self.use_reliability_gate = use_reliability_gate
        self.gate_type = gate_type
        self.downsample_ratio = downsample_ratio
        self.residual_loss_weight = residual_loss_weight
        self.gate_loss_weight = gate_loss_weight
        self.proxy_gate_min = proxy_gate_min
        self.proxy_gate_scale = proxy_gate_scale
        self.proxy_gate_power = proxy_gate_power
        self.debug_shape = debug_shape
        self._shape_printed = False
        self.last_gate = None
        self.last_gated_residual = None
        self.last_debug_stats = None
        assert gate_type in ('scalar', 'feature', 'proxy'), gate_type

        cam_channels, lidar_channels = in_channels
        self.cam_proj = nn.Conv2d(cam_channels, hidden_dim, kernel_size=1)
        self.lidar_proj = nn.Conv2d(lidar_channels, hidden_dim, kernel_size=1)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.correction = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, 1),
        )

        # Start as a gentle residual branch even if enabled.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        nn.init.zeros_(self.feature_gate[-1].weight)
        nn.init.constant_(self.feature_gate[-1].bias, float(gate_init))

    def _flatten_bev(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2).contiguous()

    def _unflatten_bev(self, x: torch.Tensor, h: int,
                       w: int) -> torch.Tensor:
        return x.transpose(1, 2).reshape(x.shape[0], -1, h, w).contiguous()

    def _feature_gate(self, cam: torch.Tensor,
                      lidar: torch.Tensor) -> torch.Tensor:
        cam_abs = cam.abs()
        lidar_abs = lidar.abs()
        stats = torch.cat([
            cam_abs.mean(dim=(2, 3)),
            cam_abs.std(dim=(2, 3), unbiased=False),
            lidar_abs.mean(dim=(2, 3)),
            lidar_abs.std(dim=(2, 3), unbiased=False),
        ], dim=1)
        gate = torch.sigmoid(self.feature_gate(stats.float()))
        return gate.view(gate.shape[0], 1, 1, 1)

    def _proxy_gate(self, reliability: dict,
                    base_feature: torch.Tensor) -> torch.Tensor:
        cam_rel = reliability['camera'].to(
            device=base_feature.device, dtype=torch.float32).view(-1, 1, 1, 1)
        lidar_rel = reliability['lidar'].to(
            device=base_feature.device, dtype=torch.float32).view(-1, 1, 1, 1)
        cam_degradation = 1.0 - cam_rel.clamp(0.0, 1.0)
        lidar_degradation = 1.0 - lidar_rel.clamp(0.0, 1.0)
        degradation = torch.maximum(cam_degradation, lidar_degradation)
        if self.proxy_gate_power != 1.0:
            degradation = degradation.clamp_min(0.0).pow(self.proxy_gate_power)
        gate = self.proxy_gate_min + self.proxy_gate_scale * degradation
        return gate.clamp(0.0, 1.0).to(dtype=base_feature.dtype)

    def forward(self, inputs: List[torch.Tensor],
                base_feature: torch.Tensor,
                reliability: dict | None = None) -> torch.Tensor:
        if not self.use_attention_fusion:
            self.last_gate = None
            self.last_gated_residual = None
            self.last_debug_stats = None
            return base_feature

        camera_bev, lidar_bev = inputs
        if self.debug_shape and not self._shape_printed:
            print('[ReliabilityGuidedAttentionResidualFusion] '
                  f'camera={tuple(camera_bev.shape)}, '
                  f'lidar={tuple(lidar_bev.shape)}, '
                  f'base={tuple(base_feature.shape)}')
            self._shape_printed = True

        out_h, out_w = base_feature.shape[-2:]
        cam = self.cam_proj(camera_bev)
        lidar = self.lidar_proj(lidar_bev)

        if self.downsample_ratio > 1:
            cam = F.avg_pool2d(cam, self.downsample_ratio,
                               self.downsample_ratio)
            lidar = F.avg_pool2d(lidar, self.downsample_ratio,
                                 self.downsample_ratio)

        h, w = lidar.shape[-2:]
        q = self._flatten_bev(lidar)
        k = self._flatten_bev(cam)
        v = k

        cross, _ = self.cross_attn(q, k, v, need_weights=False)
        tokens = self.norm_cross(q + cross)

        if self.use_self_attention:
            self_out, _ = self.self_attn(tokens, tokens, tokens,
                                         need_weights=False)
            tokens = self.norm_self(tokens + self_out)

        correction = self._unflatten_bev(tokens, h, w)
        correction = F.interpolate(
            correction, size=(out_h, out_w), mode='bilinear',
            align_corners=False)
        correction = self.correction(correction)

        if self.use_reliability_gate:
            if self.gate_type == 'proxy':
                if reliability is None:
                    raise ValueError(
                        'gate_type="proxy" requires reliability inputs.')
                gate = self._proxy_gate(reliability, base_feature)
            elif self.gate_type == 'feature':
                gate = self._feature_gate(cam, lidar).to(base_feature.dtype)
            else:
                gate = torch.sigmoid(self.gate_logit).to(base_feature.dtype)
        else:
            gate = base_feature.new_tensor(1.0)
        self.last_gate = gate.detach()
        gated_residual = gate * correction
        self.last_gated_residual = gated_residual
        base_float = base_feature.detach().float()
        residual_float = gated_residual.detach().float()
        correction_float = correction.detach().float()
        eps = 1e-6
        self.last_debug_stats = {
            'gate_mean': float(gate.detach().float().mean().cpu()),
            'gate_min': float(gate.detach().float().min().cpu()),
            'gate_max': float(gate.detach().float().max().cpu()),
            'correction_abs_mean':
            float(correction_float.abs().mean().cpu()),
            'residual_abs_mean': float(residual_float.abs().mean().cpu()),
            'base_abs_mean': float(base_float.abs().mean().cpu()),
            'residual_to_base_abs':
            float((residual_float.abs().mean() /
                   (base_float.abs().mean() + eps)).cpu()),
            'residual_to_base_l2':
            float((residual_float.pow(2).mean().sqrt() /
                   (base_float.pow(2).mean().sqrt() + eps)).cpu()),
        }
        if reliability is not None:
            self.last_debug_stats.update({
                'camera_reliability_mean':
                float(reliability['camera'].detach().float().mean().cpu()),
                'lidar_reliability_mean':
                float(reliability['lidar'].detach().float().mean().cpu()),
            })
        return base_feature + gated_residual

    def regularization_losses(self) -> dict:
        losses = {}
        if (self.residual_loss_weight > 0
                and self.last_gated_residual is not None):
            losses['loss_attention_residual'] = (
                self.last_gated_residual.float().pow(2).mean()
                * self.residual_loss_weight)
        if self.gate_loss_weight > 0 and self.last_gate is not None:
            losses['loss_attention_gate'] = (
                self.last_gate.float().mean() * self.gate_loss_weight)
        return losses
