# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from .blazeface import BlazeFace
from .onnx_model import FaceMesh
from .tessellation import FACEMESH_TESSELATION_FULL, FACEMESH_TESSELATION_PARTIAL

# ONNX inference is torch-free. The PyTorch reference nets need torch, so import
# them from the submodule directly: `from models.model import FaceMeshNet`. ROI /
# warp helpers and the size constants live in models.onnx_model if you need them.

__all__ = [
    'FACEMESH_TESSELATION_FULL',
    'FACEMESH_TESSELATION_PARTIAL',
    'BlazeFace',
    'FaceMesh',
]
