# Copyright (c) OpenMMLab. All rights reserved.
"""Synthetic sensor corruptions for BEVFusion robustness stress tests."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Sequence

import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


_SEVERITY_LEVELS = {
    'mild': 1,
    'moderate': 2,
    'severe': 3,
}


def _severity_id(severity: str) -> int:
    if severity not in _SEVERITY_LEVELS:
        raise ValueError(
            f'Unsupported severity={severity!r}. '
            f'Expected one of {tuple(_SEVERITY_LEVELS)}.')
    return _SEVERITY_LEVELS[severity]


def _stable_seed(sample_idx: Any, salt: str, base_seed: int) -> int:
    key = f'{sample_idx}_{salt}_{base_seed}'.encode('utf-8')
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def _clip_img(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.float32)


@TRANSFORMS.register_module()
class BEVFusionCameraCorruption(BaseTransform):
    """Apply deterministic multi-view image corruptions in the data pipeline.

    Args:
        corruption_type: One of ``low_light``, ``contrast``, ``fog``,
            ``gaussian_noise``, ``gaussian_blur``, ``occlusion`` or ``all``.
        severity: ``mild``, ``moderate`` or ``severe``.
        seed: Base seed. The actual RNG seed also uses ``sample_idx`` and view
            index so repeated evaluation stays deterministic.
    """

    def __init__(self,
                 corruption_type: str = 'fog',
                 severity: str = 'mild',
                 seed: int = 0,
                 apply_prob: float = 1.0) -> None:
        self.corruption_type = corruption_type
        self.severity = severity
        self.level = _severity_id(severity)
        self.seed = seed
        self.apply_prob = apply_prob

    def _rng(self, results: Dict[str, Any], view_idx: int) -> np.random.Generator:
        sample_idx = results.get('sample_idx', 'unknown')
        return np.random.default_rng(
            _stable_seed(sample_idx, f'camera_{view_idx}', self.seed))

    def _low_light(self, img: np.ndarray) -> np.ndarray:
        factors = {1: 0.75, 2: 0.55, 3: 0.35}
        return img * factors[self.level]

    def _contrast(self, img: np.ndarray) -> np.ndarray:
        factors = {1: 0.75, 2: 0.50, 3: 0.30}
        mean = img.mean(axis=(0, 1), keepdims=True)
        return (img - mean) * factors[self.level] + mean

    def _fog(self, img: np.ndarray) -> np.ndarray:
        alphas = {1: 0.18, 2: 0.35, 3: 0.55}
        return img * (1.0 - alphas[self.level]) + 255.0 * alphas[self.level]

    def _gaussian_noise(self, img: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
        sigmas = {1: 8.0, 2: 18.0, 3: 32.0}
        return img + rng.normal(0.0, sigmas[self.level], size=img.shape)

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        kernels = {1: 3, 2: 5, 3: 9}
        ksize = kernels[self.level]
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)

    def _occlusion(self, img: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        h, w = img.shape[:2]
        out = img.copy()
        counts = {1: 2, 2: 4, 3: 7}
        area_fracs = {1: 0.03, 2: 0.06, 3: 0.10}
        rect_area = max(1, int(h * w * area_fracs[self.level]))
        for _ in range(counts[self.level]):
            aspect = rng.uniform(0.5, 2.0)
            rect_h = max(1, int(np.sqrt(rect_area / aspect)))
            rect_w = max(1, int(rect_h * aspect))
            y0 = int(rng.integers(0, max(1, h - rect_h + 1)))
            x0 = int(rng.integers(0, max(1, w - rect_w + 1)))
            fill = rng.uniform(0, 45, size=(1, 1, img.shape[2]))
            out[y0:y0 + rect_h, x0:x0 + rect_w] = fill
        return out

    def _apply_one(self, img: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        ctype = self.corruption_type
        if ctype == 'low_light':
            img = self._low_light(img)
        elif ctype == 'contrast':
            img = self._contrast(img)
        elif ctype == 'fog':
            img = self._fog(img)
        elif ctype == 'gaussian_noise':
            img = self._gaussian_noise(img, rng)
        elif ctype == 'gaussian_blur':
            img = self._gaussian_blur(img)
        elif ctype == 'occlusion':
            img = self._occlusion(img, rng)
        elif ctype == 'all':
            img = self._low_light(img)
            img = self._contrast(img)
            img = self._fog(img)
            img = self._gaussian_blur(img)
            img = self._gaussian_noise(img, rng)
        else:
            raise ValueError(f'Unsupported camera corruption: {ctype}')
        return _clip_img(img)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        sample_idx = results.get('sample_idx', 'unknown')
        apply_rng = np.random.default_rng(
            _stable_seed(sample_idx, 'camera_apply', self.seed))
        if apply_rng.random() >= self.apply_prob:
            return results
        results['img'] = [
            self._apply_one(img.astype(np.float32), self._rng(results, idx))
            for idx, img in enumerate(results['img'])
        ]
        results['camera_corruption'] = dict(
            type=self.corruption_type, severity=self.severity)
        return results


@TRANSFORMS.register_module()
class BEVFusionLiDARCorruption(BaseTransform):
    """Apply deterministic point-cloud corruptions in the data pipeline.

    Args:
        corruption_type: One of ``random_dropout``, ``distance_dropout``,
            ``intensity_noise``, ``noisy_points`` or ``all``.
        severity: ``mild``, ``moderate`` or ``severe``.
        seed: Base seed used together with ``sample_idx`` for deterministic
            evaluation.
        point_cloud_range: Range used when synthetic noisy points are added.
    """

    def __init__(self,
                 corruption_type: str = 'random_dropout',
                 severity: str = 'mild',
                 seed: int = 0,
                 apply_prob: float = 1.0,
                 point_cloud_range: Sequence[float] = (-54.0, -54.0, -5.0,
                                                       54.0, 54.0, 3.0)
                 ) -> None:
        self.corruption_type = corruption_type
        self.severity = severity
        self.level = _severity_id(severity)
        self.seed = seed
        self.apply_prob = apply_prob
        self.point_cloud_range = tuple(point_cloud_range)

    def _rng(self, results: Dict[str, Any]) -> np.random.Generator:
        sample_idx = results.get('sample_idx', 'unknown')
        return np.random.default_rng(_stable_seed(sample_idx, 'lidar',
                                                  self.seed))

    def _keep_mask(self, n: int, keep_prob: torch.Tensor,
                   rng: np.random.Generator, device: torch.device) -> torch.Tensor:
        random_values = torch.from_numpy(rng.random(n)).to(device=device)
        return random_values < keep_prob

    def _random_dropout(self, tensor: torch.Tensor,
                        rng: np.random.Generator) -> torch.Tensor:
        rates = {1: 0.15, 2: 0.35, 3: 0.55}
        keep_prob = torch.full((tensor.shape[0], ), 1.0 - rates[self.level],
                               device=tensor.device)
        mask = self._keep_mask(tensor.shape[0], keep_prob, rng, tensor.device)
        return tensor[mask]

    def _distance_dropout(self, tensor: torch.Tensor,
                          rng: np.random.Generator) -> torch.Tensor:
        max_rates = {1: 0.20, 2: 0.45, 3: 0.70}
        dist = torch.linalg.norm(tensor[:, :2], dim=1)
        scaled = torch.clamp(dist / 54.0, min=0.0, max=1.0)
        drop_prob = max_rates[self.level] * scaled
        keep_prob = 1.0 - drop_prob
        mask = self._keep_mask(tensor.shape[0], keep_prob, rng, tensor.device)
        return tensor[mask]

    def _intensity_noise(self, tensor: torch.Tensor,
                         rng: np.random.Generator) -> torch.Tensor:
        if tensor.shape[1] < 4:
            return tensor
        sigmas = {1: 0.03, 2: 0.08, 3: 0.15}
        noise = torch.from_numpy(
            rng.normal(0.0, sigmas[self.level],
                       size=(tensor.shape[0], ))).to(
                           device=tensor.device, dtype=tensor.dtype)
        out = tensor.clone()
        out[:, 3] = torch.clamp(out[:, 3] + noise, min=0.0, max=1.0)
        return out

    def _noisy_points(self, tensor: torch.Tensor,
                      rng: np.random.Generator) -> torch.Tensor:
        ratios = {1: 0.03, 2: 0.07, 3: 0.12}
        n_noise = int(tensor.shape[0] * ratios[self.level])
        if n_noise <= 0:
            return tensor
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        noise_np = np.zeros((n_noise, tensor.shape[1]), dtype=np.float32)
        noise_np[:, 0] = rng.uniform(x_min, x_max, size=n_noise)
        noise_np[:, 1] = rng.uniform(y_min, y_max, size=n_noise)
        noise_np[:, 2] = rng.uniform(z_min, z_max, size=n_noise)
        if tensor.shape[1] > 3:
            noise_np[:, 3] = rng.uniform(0.0, 0.2, size=n_noise)
        noise = torch.as_tensor(noise_np, device=tensor.device,
                                dtype=tensor.dtype)
        return torch.cat([tensor, noise], dim=0)

    def _apply_tensor(self, tensor: torch.Tensor,
                      rng: np.random.Generator) -> torch.Tensor:
        ctype = self.corruption_type
        if ctype == 'random_dropout':
            tensor = self._random_dropout(tensor, rng)
        elif ctype == 'distance_dropout':
            tensor = self._distance_dropout(tensor, rng)
        elif ctype == 'intensity_noise':
            tensor = self._intensity_noise(tensor, rng)
        elif ctype == 'noisy_points':
            tensor = self._noisy_points(tensor, rng)
        elif ctype == 'all':
            tensor = self._distance_dropout(tensor, rng)
            tensor = self._intensity_noise(tensor, rng)
            tensor = self._noisy_points(tensor, rng)
        else:
            raise ValueError(f'Unsupported LiDAR corruption: {ctype}')
        return tensor

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        sample_idx = results.get('sample_idx', 'unknown')
        apply_rng = np.random.default_rng(
            _stable_seed(sample_idx, 'lidar_apply', self.seed))
        if apply_rng.random() >= self.apply_prob:
            return results
        points = results['points']
        rng = self._rng(results)
        tensor = self._apply_tensor(points.tensor, rng)
        results['points'] = points.new_point(tensor)
        results['lidar_corruption'] = dict(
            type=self.corruption_type, severity=self.severity)
        return results


@TRANSFORMS.register_module()
class BEVFusionSensorReliabilityProxy(BaseTransform):
    """Compute raw sensor reliability proxies before model preprocessing.

    This transform should run after optional camera/LiDAR corruption and before
    packing. It does not use corruption labels; it only reads the current image
    pixels and point tensor.
    """

    def __init__(self,
                 lidar_count_ref: float = 300000.0,
                 image_contrast_ref: float = 65.0,
                 image_sharpness_ref: float = 18.0) -> None:
        self.lidar_count_ref = lidar_count_ref
        self.image_contrast_ref = image_contrast_ref
        self.image_sharpness_ref = image_sharpness_ref

    def _camera_proxy(self, imgs: Sequence[np.ndarray]) -> Dict[str, float]:
        stacked = np.stack([img.astype(np.float32) for img in imgs], axis=0)
        brightness = float(stacked.mean())
        contrast = float(stacked.std())
        gray = stacked.mean(axis=-1)
        dx = float(np.abs(gray[:, :, 1:] - gray[:, :, :-1]).mean())
        dy = float(np.abs(gray[:, 1:, :] - gray[:, :-1, :]).mean())
        sharpness = 0.5 * (dx + dy)

        brightness_score = np.clip(1.0 - abs(brightness - 120.0) / 150.0,
                                   0.0, 1.0)
        contrast_score = np.clip(contrast / self.image_contrast_ref, 0.0, 1.0)
        sharpness_score = np.clip(sharpness / self.image_sharpness_ref, 0.0,
                                  1.0)
        reliability = (
            0.20 * brightness_score + 0.70 * contrast_score +
            0.10 * sharpness_score)
        return dict(
            camera_reliability_proxy=float(np.clip(reliability, 0.0, 1.0)),
            camera_brightness=float(brightness),
            camera_contrast=float(contrast),
            camera_sharpness=float(sharpness))

    def _lidar_proxy(self, points: Any) -> Dict[str, float]:
        tensor = points.tensor
        point_count = float(tensor.shape[0])
        reliability = np.clip(point_count / self.lidar_count_ref, 0.0, 1.0)
        return dict(
            lidar_reliability_proxy=float(reliability),
            lidar_point_count=float(point_count))

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'img' in results:
            results.update(self._camera_proxy(results['img']))
        if 'points' in results:
            results.update(self._lidar_proxy(results['points']))
        return results
