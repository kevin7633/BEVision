# Copyright (c) OpenMMLab. All rights reserved.
"""Sensor corruptions and reliability proxy transforms for BEVFusion."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


_SEVERITY = {
    'mild': 1,
    'moderate': 2,
    'severe': 3,
}


def _level(severity: str) -> int:
    if severity not in _SEVERITY:
        raise ValueError(f'Unsupported severity={severity!r}.')
    return _SEVERITY[severity]


def _stable_seed(sample_idx: Any, salt: str, seed: int) -> int:
    key = f'{sample_idx}_{salt}_{seed}'.encode('utf-8')
    return int(hashlib.sha1(key).hexdigest()[:8], 16)


def _clip_img(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 255.0).astype(np.float32)


@TRANSFORMS.register_module()
class BEVFusionCameraCorruption(BaseTransform):
    """Deterministic camera corruptions for robustness evaluation.

    Supported types are ``brightness_down``, ``contrast_down``, ``blur``,
    ``dropout_camera``, ``fog`` and ``all``.
    """

    def __init__(self,
                 corruption_type: str = 'brightness_down',
                 severity: str = 'mild',
                 seed: int = 0,
                 apply_prob: float = 1.0) -> None:
        self.corruption_type = corruption_type
        self.severity = severity
        self.level = _level(severity)
        self.seed = seed
        self.apply_prob = apply_prob

    def _rng(self, results: Dict[str, Any],
             view_idx: int) -> np.random.Generator:
        sample_idx = results.get('sample_idx', 'unknown')
        return np.random.default_rng(
            _stable_seed(sample_idx, f'camera_{view_idx}', self.seed))

    def _brightness_down(self, img: np.ndarray) -> np.ndarray:
        return img * {1: 0.75, 2: 0.55, 3: 0.35}[self.level]

    def _contrast_down(self, img: np.ndarray) -> np.ndarray:
        factor = {1: 0.75, 2: 0.50, 3: 0.30}[self.level]
        mean = img.mean(axis=(0, 1), keepdims=True)
        return (img - mean) * factor + mean

    def _blur(self, img: np.ndarray) -> np.ndarray:
        kernel = {1: 3, 2: 5, 3: 9}[self.level]
        return cv2.GaussianBlur(img, (kernel, kernel), sigmaX=0)

    def _fog(self, img: np.ndarray) -> np.ndarray:
        alpha = {1: 0.18, 2: 0.35, 3: 0.55}[self.level]
        return img * (1.0 - alpha) + 255.0 * alpha

    def _apply_one(self, img: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        kind = self.corruption_type
        if kind == 'brightness_down':
            img = self._brightness_down(img)
        elif kind == 'contrast_down':
            img = self._contrast_down(img)
        elif kind == 'blur':
            img = self._blur(img)
        elif kind == 'fog':
            img = self._fog(img)
        elif kind == 'dropout_camera':
            # View-level dropout is handled in transform() so all pixels in the
            # selected camera are removed consistently.
            pass
        elif kind == 'all':
            img = self._brightness_down(img)
            img = self._contrast_down(img)
            img = self._blur(img)
            img = self._fog(img)
        else:
            raise ValueError(f'Unsupported camera corruption: {kind}')
        return _clip_img(img)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        sample_idx = results.get('sample_idx', 'unknown')
        apply_rng = np.random.default_rng(
            _stable_seed(sample_idx, 'camera_apply', self.seed))
        if apply_rng.random() >= self.apply_prob:
            return results

        imgs = [img.astype(np.float32) for img in results['img']]
        if self.corruption_type == 'dropout_camera':
            counts = {1: 1, 2: 2, 3: 3}
            view_count = len(imgs)
            drop_num = min(counts[self.level], view_count)
            chosen = apply_rng.choice(view_count, size=drop_num, replace=False)
            for idx in chosen:
                imgs[int(idx)] = np.zeros_like(imgs[int(idx)])
        else:
            imgs = [
                self._apply_one(img, self._rng(results, idx))
                for idx, img in enumerate(imgs)
            ]
        results['img'] = imgs
        results['camera_corruption'] = dict(
            type=self.corruption_type, severity=self.severity)
        return results


@TRANSFORMS.register_module()
class BEVFusionLiDARCorruption(BaseTransform):
    """Deterministic LiDAR corruptions for robustness evaluation."""

    def __init__(self,
                 corruption_type: str = 'point_dropout',
                 severity: str = 'mild',
                 seed: int = 0,
                 apply_prob: float = 1.0) -> None:
        self.corruption_type = corruption_type
        self.severity = severity
        self.level = _level(severity)
        self.seed = seed
        self.apply_prob = apply_prob

    def _rng(self, results: Dict[str, Any]) -> np.random.Generator:
        sample_idx = results.get('sample_idx', 'unknown')
        return np.random.default_rng(
            _stable_seed(sample_idx, 'lidar', self.seed))

    def _keep_mask(self, n: int, keep_prob: torch.Tensor,
                   rng: np.random.Generator,
                   device: torch.device) -> torch.Tensor:
        random_values = torch.from_numpy(rng.random(n)).to(device=device)
        return random_values < keep_prob

    def _point_dropout(self, points: torch.Tensor,
                       rng: np.random.Generator) -> torch.Tensor:
        rates = {1: 0.15, 2: 0.35, 3: 0.55}
        keep_prob = torch.full((points.shape[0], ),
                               1.0 - rates[self.level],
                               device=points.device)
        return points[self._keep_mask(points.shape[0], keep_prob, rng,
                                      points.device)]

    def _distance_dropout(self, points: torch.Tensor,
                          rng: np.random.Generator) -> torch.Tensor:
        max_rates = {1: 0.20, 2: 0.45, 3: 0.70}
        distance = torch.linalg.norm(points[:, :2], dim=1)
        drop_prob = max_rates[self.level] * (distance / 54.0).clamp(0.0, 1.0)
        keep_prob = 1.0 - drop_prob
        return points[self._keep_mask(points.shape[0], keep_prob, rng,
                                      points.device)]

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        sample_idx = results.get('sample_idx', 'unknown')
        apply_rng = np.random.default_rng(
            _stable_seed(sample_idx, 'lidar_apply', self.seed))
        if apply_rng.random() >= self.apply_prob:
            return results
        points = results['points']
        tensor = points.tensor
        rng = self._rng(results)
        if self.corruption_type == 'point_dropout':
            tensor = self._point_dropout(tensor, rng)
        elif self.corruption_type == 'distance_dropout':
            tensor = self._distance_dropout(tensor, rng)
        elif self.corruption_type == 'all':
            tensor = self._distance_dropout(tensor, rng)
            tensor = self._point_dropout(tensor, rng)
        else:
            raise ValueError(f'Unsupported LiDAR corruption: '
                             f'{self.corruption_type}')
        results['points'] = points.new_point(tensor)
        results['lidar_corruption'] = dict(
            type=self.corruption_type, severity=self.severity)
        return results


@TRANSFORMS.register_module()
class BEVFusionSensorReliability(BaseTransform):
    """Compute bounded image and LiDAR reliability proxies.

    The transform runs before packing so image reliability is computed on pixel
    values before normalization. It records both per-camera scores and a
    scene-level average in the metainfo.
    """

    def __init__(self,
                 brightness_target: float = 127.5,
                 brightness_tolerance: float = 127.5,
                 contrast_ref: float = 65.0,
                 sharpness_ref: float = 120.0,
                 lidar_count_ref: float = 300000.0,
                 image_weights: Sequence[float] = (0.25, 0.45, 0.30),
                 compute_lidar_bev_map: bool = False,
                 bev_map_size: Sequence[int] = (32, 32),
                 point_cloud_range: Sequence[float] = (-54.0, -54.0, -5.0,
                                                       54.0, 54.0, 3.0)
                 ) -> None:
        self.brightness_target = brightness_target
        self.brightness_tolerance = brightness_tolerance
        self.contrast_ref = contrast_ref
        self.sharpness_ref = sharpness_ref
        self.lidar_count_ref = lidar_count_ref
        self.image_weights = tuple(image_weights)
        self.compute_lidar_bev_map = compute_lidar_bev_map
        self.bev_map_size = tuple(bev_map_size)
        self.point_cloud_range = tuple(point_cloud_range)

    def _camera_scores(self, imgs: Sequence[np.ndarray]) -> Dict[str, Any]:
        per_view: List[float] = []
        brightness_values: List[float] = []
        contrast_values: List[float] = []
        sharpness_values: List[float] = []
        wb, wc, ws = self.image_weights
        for img in imgs:
            img_f = img.astype(np.float32)
            gray = cv2.cvtColor(_clip_img(img_f).astype(np.uint8),
                                cv2.COLOR_BGR2GRAY).astype(np.float32)
            brightness = float(gray.mean())
            contrast = float(gray.std())
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            brightness_score = np.clip(
                1.0 - abs(brightness - self.brightness_target) /
                self.brightness_tolerance, 0.0, 1.0)
            contrast_score = np.clip(contrast / self.contrast_ref, 0.0, 1.0)
            sharpness_score = np.clip(sharpness / self.sharpness_ref, 0.0,
                                      1.0)
            reliability = wb * brightness_score + wc * contrast_score + ws * sharpness_score
            per_view.append(float(np.clip(reliability, 0.0, 1.0)))
            brightness_values.append(brightness)
            contrast_values.append(contrast)
            sharpness_values.append(sharpness)

        return dict(
            image_reliability=float(np.mean(per_view)),
            image_reliability_per_view=per_view,
            image_brightness_per_view=brightness_values,
            image_contrast_per_view=contrast_values,
            image_sharpness_per_view=sharpness_values)

    def _lidar_bev_map(self, tensor: torch.Tensor) -> np.ndarray:
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        points = tensor[:, :2].detach().cpu().numpy()
        hist, _, _ = np.histogram2d(
            points[:, 1],
            points[:, 0],
            bins=self.bev_map_size,
            range=[[y_min, y_max], [x_min, x_max]])
        ref = max(float(hist.mean()) * 2.0, 1.0)
        return np.clip(hist / ref, 0.0, 1.0).astype(np.float32)

    def _lidar_scores(self, points: Any) -> Dict[str, Any]:
        tensor = points.tensor
        point_count = float(tensor.shape[0])
        out = dict(
            lidar_reliability=float(
                np.clip(point_count / self.lidar_count_ref, 0.0, 1.0)),
            lidar_point_count=point_count)
        if self.compute_lidar_bev_map and tensor.shape[0] > 0:
            out['lidar_reliability_map'] = self._lidar_bev_map(tensor)
        return out

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'img' in results:
            results.update(self._camera_scores(results['img']))
        if 'points' in results:
            results.update(self._lidar_scores(results['points']))
        return results
