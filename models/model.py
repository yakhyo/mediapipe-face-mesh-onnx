# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ['BlazeFaceNet', 'FaceLandmarkerNet', 'FaceMeshNet']

# BatchNorm does not appear here because the released weights have it folded
# into the convolutions — verified: Google's own .tflite files contain no
# normalization ops (TFLite always folds BN on export), which is why the convs
# carry bias. The folding cannot be inverted, so the original BN statistics are
# unrecoverable. Structures were recovered from this repo's verified ONNX
# graphs, and the shipped weights were mapped from them and checked for parity
# against ONNX Runtime.


class MeshBlock(nn.Module):
    """Residual depthwise-separable block of Face Mesh: DW 3x3 -> PW 1x1, skip, PReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            # pw-linear
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.act = nn.PReLU(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv(x))


class MeshDown(nn.Module):
    """Stride-2 transition of Face Mesh: DW 3x3 s2 -> PW 1x1, MaxPool skip, PReLU.

    The depthwise uses TF-style 'same' padding (bottom/right only); the skip is
    zero-padded along channels when the block widens.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            # dw
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels),
            # pw-linear
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.extra_channels = out_channels - in_channels
        self.act = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        skip = self.pool(x)
        if self.extra_channels > 0:
            skip = F.pad(skip, (0, 0, 0, 0, 0, self.extra_channels))
        return self.act(self.conv(x) + skip)


class FaceMeshNet(nn.Module):
    """MediaPipe Face Mesh: 468 dense 3D landmarks from a 192x192 face crop.

    Input is (N, 3, 192, 192) RGB in [0, 1]. Returns screen landmarks
    (N, 468, 3) in 192-crop pixels and the face-presence logit (N, 1) —
    apply sigmoid.

    Example:
        >>> model = FaceMeshNet()
        >>> model.load_state_dict(torch.load('weights/face_mesh_192x192.pt'))
        >>> landmarks, score = model(x)

    Reference:
        https://arxiv.org/abs/1907.06724
        https://github.com/google-ai-edge/mediapipe
    """

    def __init__(self) -> None:
        super().__init__()

        feature_setting = [
            # c, n — channels and number of residual blocks per stage
            [16, 2],
            [32, 2],
            [64, 2],
            [128, 2],
            [128, 2],
        ]

        features: list[nn.Module] = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.PReLU(16),
        ]
        in_channels = 16
        for stage, (channels, num_blocks) in enumerate(feature_setting):
            if stage > 0:
                features.append(MeshDown(in_channels, channels))
            features.extend(MeshBlock(channels) for _ in range(num_blocks))
            in_channels = channels
        self.features = nn.Sequential(*features)

        self.landmark_head = nn.Sequential(
            MeshDown(128, 128),
            MeshBlock(128),
            MeshBlock(128),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.PReLU(32),
            MeshBlock(32),
            nn.Conv2d(32, 1404, kernel_size=3, stride=3),
        )
        self.score_head = nn.Sequential(
            MeshDown(128, 128),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.PReLU(32),
            MeshBlock(32),
            nn.Conv2d(32, 1, kernel_size=3, stride=3),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.features(x)

        landmarks = self.landmark_head(x)
        score = self.score_head(x)
        return landmarks.reshape(-1, 468, 3), score.reshape(-1, 1)


class LandmarkerBlock(nn.Module):
    """Residual inverted bottleneck of Face Landmarker: PW-reduce -> DW 3x3 -> PW-expand, skip, PReLU.

    Narrow in the middle, unlike `MeshBlock`, which keeps full width through a
    DW -> PW pair. The activation sits inside the bottleneck, between the reducing
    1x1 and the depthwise; there is none after the depthwise.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.conv = nn.Sequential(
            # pw-reduce
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.PReLU(hidden),
            # dw
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            # pw-expand
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.act = nn.PReLU(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv(x))


class LandmarkerDown(nn.Module):
    """Stride-2 transition of Face Landmarker: 2x2 s2 -> DW 3x3 -> PW 1x1, MaxPool skip, PReLU.

    The 2x2 stride-2 convolution opens the bottleneck instead of a depthwise, and
    because every resolution here is even it tiles exactly — no padding, unlike the
    TF-style 'same' padding `MeshDown` needs. The skip is zero-padded along channels
    when the block widens.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=2, stride=2),
            nn.PReLU(hidden),
            # dw
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            # pw-expand
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.extra_channels = out_channels - in_channels
        self.act = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        skip = self.pool(x)
        if self.extra_channels > 0:
            skip = F.pad(skip, (0, 0, 0, 0, 0, self.extra_channels))
        return self.act(self.conv(x) + skip)


class FaceLandmarkerNet(nn.Module):
    """MediaPipe Face Landmarker: 478 dense 3D landmarks from a 256x256 face crop.

    The successor to `FaceMeshNet`. Where MediaPipe's earlier 478-point model bolted
    attention branches onto the 468-point net — three custom TFLite ops and a five-way
    merge outside the graph — this one is a single network that predicts all 478 points
    directly. Landmarks 0-467 keep the Face Mesh topology and ordering, so the same
    tessellation draws it; 468-477 are the irises.

    Input is (N, 3, 256, 256) RGB in [0, 1]. Returns screen landmarks (N, 478, 3) in
    256-crop pixels and the face-presence logit (N, 1) — apply sigmoid.

    The graph carries a second, already-sigmoided scalar head alongside that logit.
    Google's MediaPipe FaceMesh V2 model card lists three outputs — the landmarks, the
    face flag, and "a limited set of blendshapes that includes cheekPuff and tongueOut",
    meant to be consumed by the separate Blendshapes model — so that head is most likely
    the blendshape one. The landmarks do not depend on it, so `conv_presence` is kept
    here for weight fidelity but left out of `forward`, which preserves the two-output
    contract `FaceMeshNet` established and the ONNX consumers rely on.

    Model card: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf

    Example:
        >>> model = FaceLandmarkerNet()
        >>> model.load_state_dict(torch.load('weights/face_landmarker_256x256.pt'))
        >>> landmarks, score = model(x)

    Reference:
        https://github.com/google-ai-edge/mediapipe
    """

    NUM_LANDMARKS = 478

    def __init__(self) -> None:
        super().__init__()

        feature_setting = [
            # c, n — channels and number of residual blocks per stage
            [16, 4],
            [32, 4],
            [64, 4],
            [128, 4],
            [128, 4],
            [128, 4],
            [128, 4],
        ]

        features: list[nn.Module] = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.PReLU(16),
        ]
        in_channels = 16
        for stage, (channels, num_blocks) in enumerate(feature_setting):
            if stage > 0:
                features.append(LandmarkerDown(in_channels, channels))
            features.extend(LandmarkerBlock(channels) for _ in range(num_blocks))
            in_channels = channels
        self.features = nn.Sequential(*features)

        # Declared in the order the source .tflite emits them, which is the order the
        # weights were mapped in.
        self.conv_presence = nn.Conv2d(128, 1, kernel_size=2)
        self.conv_score = nn.Conv2d(128, 1, kernel_size=2)
        self.conv_landmarks = nn.Conv2d(128, 3 * self.NUM_LANDMARKS, kernel_size=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.features(x)

        landmarks = self.conv_landmarks(x)
        score = self.conv_score(x)
        return landmarks.reshape(-1, self.NUM_LANDMARKS, 3), score.reshape(-1, 1)


class BlazeBlock(nn.Module):
    """BlazeFace building block: DW 3x3 -> PW 1x1, residual skip, ReLU.

    With stride 2 the skip becomes a 2x2 MaxPool and the depthwise uses
    TF-style 'same' padding (bottom/right only); the skip is zero-padded along
    channels when the block widens.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f'stride should be 1 or 2 instead of {stride}')
        self.stride = stride

        layers: list[nn.Module] = []
        if stride == 2:
            layers.append(nn.ZeroPad2d((0, 1, 0, 1)))
        layers.extend(
            [
                # dw
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1 if stride == 1 else 0,
                    groups=in_channels,
                ),
                # pw-linear
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.extra_channels = out_channels - in_channels
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        skip = F.max_pool2d(x, 2) if self.stride == 2 else x
        if self.extra_channels > 0:
            skip = F.pad(skip, (0, 0, 0, 0, 0, self.extra_channels))
        return self.act(self.conv(x) + skip)


class BlazeFaceNet(nn.Module):
    """BlazeFace short-range: SSD face detector on a 128x128 letterboxed image.

    Input is (N, 3, 128, 128) RGB in [-1, 1]. Returns raw box/keypoint
    regressors (N, 896, 16) and score logits (N, 896, 1) over two anchor
    grids — 16x16 with 2 anchors per cell first, then 8x8 with 6 — matching
    the anchor order of `models/blazeface.py`, where decode and weighted NMS
    live.

    Example:
        >>> model = BlazeFaceNet()
        >>> model.load_state_dict(torch.load('weights/face_detection_short_range.pt'))
        >>> regressors, logits = model(x)

    Reference:
        https://arxiv.org/abs/1907.05047
        https://github.com/google-ai-edge/mediapipe
    """

    def __init__(self) -> None:
        super().__init__()

        self.features_16 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            BlazeBlock(24, 24),
            BlazeBlock(24, 28),
            BlazeBlock(28, 32, stride=2),
            BlazeBlock(32, 36),
            BlazeBlock(36, 42),
            BlazeBlock(42, 48, stride=2),
            BlazeBlock(48, 56),
            BlazeBlock(56, 64),
            BlazeBlock(64, 72),
            BlazeBlock(72, 80),
            BlazeBlock(80, 88),
        )
        self.reg_16 = nn.Conv2d(88, 2 * 16, kernel_size=1)
        self.features_8 = nn.Sequential(
            BlazeBlock(88, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
        )
        self.reg_8 = nn.Conv2d(96, 6 * 16, kernel_size=1)
        self.cls_8 = nn.Conv2d(96, 6 * 1, kernel_size=1)
        self.cls_16 = nn.Conv2d(88, 2 * 1, kernel_size=1)

    @staticmethod
    def _flatten(x: Tensor, num_coords: int) -> Tensor:
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, num_coords)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x16 = self.features_16(x)
        x8 = self.features_8(x16)

        regressors = torch.cat([self._flatten(self.reg_16(x16), 16), self._flatten(self.reg_8(x8), 16)], dim=1)
        logits = torch.cat([self._flatten(self.cls_16(x16), 1), self._flatten(self.cls_8(x8), 1)], dim=1)
        return regressors, logits
