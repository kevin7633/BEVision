#!/usr/bin/env python
"""Smoke test for BEVFusion attention residual fusion on one mini sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--debug-shape', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    if args.debug_shape:
        cfg.model.attention_fusion_layer.debug_shape = True
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    sample = dataset[args.sample_index]
    batch = pseudo_collate([sample])
    with torch.no_grad():
        outputs = model.test_step(batch)

    pred = outputs[0].pred_instances_3d
    num_boxes = len(pred.scores_3d)
    mean_score = (
        float(pred.scores_3d.mean().cpu()) if num_boxes > 0 else 0.0
    )
    attention_layer = getattr(model, 'attention_fusion_layer', None)
    gate = None
    if attention_layer is not None and hasattr(attention_layer, 'gate_logit'):
        gate = float(torch.sigmoid(attention_layer.gate_logit).detach().cpu())
    last_gate = None
    if attention_layer is not None and getattr(attention_layer, 'last_gate',
                                              None) is not None:
        last_gate = attention_layer.last_gate.detach().float().mean().cpu()
        last_gate = float(last_gate)

    print(f'config={Path(args.config).name}')
    print(f'device={device}')
    print(f'sample_index={args.sample_index}')
    print(f'num_predictions={num_boxes}')
    print(f'mean_score={mean_score:.6f}')
    if gate is not None:
        print(f'attention_gate={gate:.8f}')
    if last_gate is not None:
        print(f'last_forward_gate_mean={last_gate:.8f}')


if __name__ == '__main__':
    main()
