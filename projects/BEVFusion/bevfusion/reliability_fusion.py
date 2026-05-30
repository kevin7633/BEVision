# Copyright (c) OpenMMLab. All rights reserved.
"""Reliability-guided residual BEV fusion modules.

The module keeps the LiDAR BEV feature as a stable base by default and adds a
bounded camera/fusion correction:

    fused = base + gate * correction
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger

from mmdet3d.registry import MODELS


class _ChannelAttention(nn.Module):
    """Small squeeze-excitation block for optional BEV channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class _SpatialAttention(nn.Module):
    """Lightweight spatial attention over BEV cells."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        max_value = x.max(dim=1, keepdim=True).values
        attn = torch.sigmoid(self.conv(torch.cat([mean, max_value], dim=1)))
        return x * attn


@MODELS.register_module()
class ReliabilityGuidedResidualFusion(nn.Module):
    """Residual correction layer controlled by sensor reliability.

    Args:
        in_channels: ``[camera_bev_channels, lidar_bev_channels]``.
        out_channels: Output BEV channel count expected by the LiDAR backbone.
        base_source: ``lidar`` uses LiDAR BEV as the residual base. ``fused``
            uses the original ConvFuser output as the base.
        correction_source: ``camera_lidar`` predicts correction from both BEV
            branches, ``camera`` uses only camera BEV, and ``fused`` uses the
            original ConvFuser output.
        gate_mode: ``image_only``, ``lidar_only``, ``image_lidar``, or
            ``learned``. ``image_lidar`` implements
            ``sigmoid(alpha * image_rel - beta * lidar_rel + bias)``.
        gate_formula: ``sigmoid`` or ``proxy_product``. The latter implements
            ``proxy_gate_min + proxy_gate_scale * image_rel * (1 - lidar_rel)``.
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 out_channels: int,
                 enabled: bool = True,
                 base_source: str = 'lidar',
                 correction_source: str = 'camera_lidar',
                 hidden_channels: int = 256,
                 use_reliability_gate: bool = True,
                 gate_mode: str = 'image_lidar',
                 gate_formula: str = 'sigmoid',
                 alpha: float = 4.0,
                 beta: float = 2.0,
                 bias: float = -1.0,
                 proxy_gate_min: float = 0.0,
                 proxy_gate_scale: float = 0.5,
                 gate_min: float = 0.0,
                 gate_max: float = 0.75,
                 learned_gate_init: float = -4.0,
                 use_channel_attention: bool = False,
                 use_spatial_attention: bool = False,
                 residual_loss_weight: float = 0.0,
                 gate_loss_weight: float = 0.0,
                 debug_shape: bool = False) -> None:
        super().__init__()
        assert len(in_channels) == 2, in_channels
        assert base_source in ('lidar', 'fused')
        assert correction_source in ('camera', 'camera_lidar', 'fused')
        assert gate_mode in ('image_only', 'lidar_only', 'image_lidar',
                             'learned')
        assert gate_formula in ('sigmoid', 'proxy_product')

        self.enabled = enabled
        self.base_source = base_source
        self.correction_source = correction_source
        self.use_reliability_gate = use_reliability_gate
        self.gate_mode = gate_mode
        self.gate_formula = gate_formula
        self.alpha = alpha
        self.beta = beta
        self.bias = bias
        self.proxy_gate_min = proxy_gate_min
        self.proxy_gate_scale = proxy_gate_scale
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.residual_loss_weight = residual_loss_weight
        self.gate_loss_weight = gate_loss_weight
        self.debug_shape = debug_shape
        self._shape_logged = False
        self.last_gate: Optional[torch.Tensor] = None
        self.last_residual: Optional[torch.Tensor] = None
        self.last_stats: Dict[str, torch.Tensor] = {}

        cam_channels, lidar_channels = in_channels
        conv_cfg = dict(type='Conv2d')
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU', inplace=True)
        self.camera_proj = ConvModule(
            cam_channels, hidden_channels, 1, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.lidar_proj = ConvModule(
            lidar_channels, hidden_channels, 1, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.fused_proj = ConvModule(
            out_channels, hidden_channels, 1, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.base_proj = (nn.Identity() if lidar_channels == out_channels else
                          nn.Conv2d(lidar_channels, out_channels, 1))

        if correction_source == 'camera_lidar':
            correction_in = hidden_channels * 2
        else:
            correction_in = hidden_channels
        self.correction = nn.Sequential(
            ConvModule(
                correction_in,
                hidden_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )
        self.channel_attention = (
            _ChannelAttention(out_channels) if use_channel_attention else
            nn.Identity())
        self.spatial_attention = (
            _SpatialAttention() if use_spatial_attention else nn.Identity())
        self.learned_gate_logit = nn.Parameter(
            torch.tensor(float(learned_gate_init)))

        # Start from the baseline/base feature and let the residual grow only
        # when optimization finds a useful correction.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def _resize_like(self, src: torch.Tensor,
                     ref: torch.Tensor) -> torch.Tensor:
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(
            src, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def _reliability_value(self, reliability: Optional[dict], key: str,
                           base: torch.Tensor,
                           default: float) -> torch.Tensor:
        if reliability is None or key not in reliability:
            value = base.new_full((base.shape[0], ), default)
        else:
            value = reliability[key].to(
                device=base.device, dtype=torch.float32).view(base.shape[0])
        return value.clamp(0.0, 1.0)

    def _make_gate(self, base: torch.Tensor,
                   reliability: Optional[dict]) -> torch.Tensor:
        if not self.use_reliability_gate:
            return base.new_ones((base.shape[0], 1, 1, 1))
        if self.gate_mode == 'learned':
            gate = torch.sigmoid(self.learned_gate_logit).expand(
                base.shape[0], 1, 1, 1)
            return gate.to(device=base.device, dtype=base.dtype)

        image_rel = self._reliability_value(reliability, 'image', base, 0.5)
        lidar_rel = self._reliability_value(reliability, 'lidar', base, 0.5)

        if self.gate_formula == 'proxy_product':
            if self.gate_mode == 'image_only':
                raw_gate = image_rel
            elif self.gate_mode == 'lidar_only':
                raw_gate = 1.0 - lidar_rel
            else:
                raw_gate = image_rel * (1.0 - lidar_rel)
            gate = self.proxy_gate_min + self.proxy_gate_scale * raw_gate
        else:
            if self.gate_mode == 'image_only':
                logit = self.alpha * image_rel + self.bias
            elif self.gate_mode == 'lidar_only':
                logit = self.alpha * (1.0 - lidar_rel) + self.bias
            else:
                logit = self.alpha * image_rel - self.beta * lidar_rel + self.bias
            gate = torch.sigmoid(logit)

        gate = gate.clamp(self.gate_min, self.gate_max)
        return gate.view(base.shape[0], 1, 1, 1).to(dtype=base.dtype)

    def _correction_input(self, camera_bev: torch.Tensor,
                          lidar_bev: torch.Tensor,
                          fused_feature: Optional[torch.Tensor],
                          base: torch.Tensor) -> torch.Tensor:
        camera = self._resize_like(self.camera_proj(camera_bev), base)
        lidar = self._resize_like(self.lidar_proj(lidar_bev), base)
        if self.correction_source == 'camera':
            return camera
        if self.correction_source == 'fused':
            if fused_feature is None:
                raise ValueError('correction_source="fused" requires '
                                 'fused_feature.')
            return self._resize_like(self.fused_proj(fused_feature), base)
        return torch.cat([camera, lidar], dim=1)

    def _update_stats(self, gate: torch.Tensor, correction: torch.Tensor,
                      residual: torch.Tensor, base: torch.Tensor,
                      reliability: Optional[dict]) -> None:
        eps = 1e-6
        stats = {
            'gate_mean': gate.detach().float().mean(),
            'gate_min': gate.detach().float().min(),
            'gate_max': gate.detach().float().max(),
            'correction_abs_mean': correction.detach().float().abs().mean(),
            'residual_abs_mean': residual.detach().float().abs().mean(),
            'base_abs_mean': base.detach().float().abs().mean(),
        }
        stats['residual_to_base_abs'] = (
            stats['residual_abs_mean'] / (stats['base_abs_mean'] + eps))
        if reliability is not None:
            for src_key, stat_key in (('image', 'image_reliability_mean'),
                                      ('lidar', 'lidar_reliability_mean')):
                if src_key in reliability:
                    stats[stat_key] = reliability[src_key].detach().float().mean()
        self.last_stats = stats

    def forward(self,
                inputs: List[torch.Tensor],
                fused_feature: Optional[torch.Tensor] = None,
                reliability: Optional[dict] = None) -> torch.Tensor:
        camera_bev, lidar_bev = inputs
        if not self.enabled:
            return fused_feature if fused_feature is not None else lidar_bev

        base = (fused_feature if self.base_source == 'fused' else
                self.base_proj(lidar_bev))
        if base is None:
            raise ValueError('base_source="fused" requires fused_feature.')

        if self.debug_shape and not self._shape_logged:
            logger = MMLogger.get_current_instance()
            logger.info(
                'ReliabilityGuidedResidualFusion shapes: '
                f'camera={tuple(camera_bev.shape)}, '
                f'lidar={tuple(lidar_bev.shape)}, base={tuple(base.shape)}')
            self._shape_logged = True

        correction_input = self._correction_input(camera_bev, lidar_bev,
                                                  fused_feature, base)
        correction = self.correction(correction_input)
        correction = self.channel_attention(correction)
        correction = self.spatial_attention(correction)
        gate = self._make_gate(base, reliability)
        residual = gate * correction

        self.last_gate = gate.detach()
        self.last_residual = residual
        self._update_stats(gate, correction, residual, base, reliability)
        return base + residual

    def regularization_losses(self) -> Dict[str, torch.Tensor]:
        losses = {}
        if self.residual_loss_weight > 0 and self.last_residual is not None:
            losses['loss_reliability_residual'] = (
                self.last_residual.float().pow(2).mean() *
                self.residual_loss_weight)
        if self.gate_loss_weight > 0 and self.last_gate is not None:
            losses['loss_reliability_gate'] = (
                self.last_gate.float().mean() * self.gate_loss_weight)
        return losses

    def log_vars(self) -> Dict[str, torch.Tensor]:
        """Return scalar tensors that MMEngine can log without adding to loss."""
        return {f'reliability/{k}': v for k, v in self.last_stats.items()}
