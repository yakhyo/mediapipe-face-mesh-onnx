# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Export the PyTorch nets (models/model.py) to ONNX.

The .pt weights are the source of truth; this produces the ONNX the inference
pipeline runs (pure network, batched, post-processing done in Python). Each
export is sanity-checked against its PyTorch reference.

Dev-only requirements (not in requirements.txt): pip install torch onnx onnxruntime

Usage: python onnx_export.py                 # both nets
       python onnx_export.py -m blazeface    # one net
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from models.model import BlazeFaceNet, FaceLandmarkerNet, FaceMeshNet

WEIGHTS = Path(__file__).parent / 'weights'

# name -> (net class, checkpoint, input shape, output names, onnx filename)
MODELS = {
    'facemesh': (
        FaceMeshNet,
        'face_mesh_192x192.pt',
        (1, 3, 192, 192),
        ['landmarks', 'score'],
        'face_mesh_Nx3x192x192.onnx',
    ),
    'facelandmarker': (
        FaceLandmarkerNet,
        'face_landmarker_256x256.pt',
        (1, 3, 256, 256),
        ['landmarks', 'score'],
        'face_landmarker_Nx3x256x256.onnx',
    ),
    'blazeface': (
        BlazeFaceNet,
        'face_detection_short_range.pt',
        (1, 3, 128, 128),
        ['regressors', 'scores'],
        'face_detection_short_range.onnx',
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export the PyTorch Face Mesh / BlazeFace nets to ONNX')
    parser.add_argument('-m', '--model', choices=[*MODELS, 'all'], default='all', help='Which net to export')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    return parser.parse_args()


@torch.no_grad()
def export(name: str, opset: int) -> None:
    net_cls, checkpoint, shape, output_names, filename = MODELS[name]
    net = net_cls().eval()
    net.load_state_dict(torch.load(WEIGHTS / checkpoint))

    target = WEIGHTS / filename
    dummy = torch.rand(*shape)

    # Batch dim (dim 0) is dynamic on the input and every output — these nets run
    # batched inference, all faces in one pass, like the shipped Nx models.
    dynamic_axes = {'input': {0: 'batch'}}
    for output_name in output_names:
        dynamic_axes[output_name] = {0: 'batch'}

    torch.onnx.export(
        net,
        dummy,
        str(target),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    # Sanity-check the export against the PyTorch reference before trusting it.
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    onnx_outputs = session.run(None, {'input': dummy.numpy()})
    max_diff = max(float(np.abs(a - b.numpy()).max()) for a, b in zip(onnx_outputs, net(dummy), strict=True))
    print(f'{name}: {target.name} ({target.stat().st_size / 1e6:.1f} MB), vs PyTorch max |diff| = {max_diff:.2e}')


def main() -> None:
    args = parse_arguments()
    names = list(MODELS) if args.model == 'all' else [args.model]
    for name in names:
        export(name, args.opset)


if __name__ == '__main__':
    main()
