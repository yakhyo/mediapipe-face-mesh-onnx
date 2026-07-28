# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""BlazeFace short-range face detection (MediaPipe's own Face Mesh seed).

The pure detector network, exported from this repo's PyTorch `BlazeFaceNet`
(models/model.py) — the same short-range detector `mp.solutions.face_mesh` runs
internally — with MediaPipe's post-processing reimplemented here: SSD anchor
decode and **weighted** NMS (overlapping candidates are score-averaged,
keypoints included, rather than discarded). It is the exact detector
`mp.solutions.face_mesh` runs, so the mesh seeds from MediaPipe-faithful boxes.

Input: 128x128 RGB letterboxed, normalized to [-1, 1] (unlike the landmark
models' [0, 1]). Raw outputs: 896 anchors x (4 box + 6 keypoints x 2) and
896 logits. Keypoint order: right eye, left eye, nose tip, mouth center,
right ear tragion, left ear tragion — rows 0/1 are therefore the viewer-left
and viewer-right eye, which this repo's `roi_from_box` uses for the roll angle.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

FACE_INPUT_SIZE = 128
NUM_FACE_KEYPOINTS = 6


def _generate_face_anchors() -> np.ndarray:
    """SSD anchor centers for face_detection_short_range at 128x128.

    MediaPipe's SsdAnchorsCalculator config: strides [8, 16, 16, 16], aspect
    ratio 1.0 plus one interpolated scale per layer, fixed_anchor_size. Layers
    sharing a stride stack their anchors per cell: 16x16x2 + 8x8x6 = 896.
    """
    anchors = []
    strides = [8, 16, 16, 16]
    idx = 0
    while idx < len(strides):
        last = idx
        while last < len(strides) and strides[last] == strides[idx]:
            last += 1
        repeats = 2 * (last - idx)
        cells = FACE_INPUT_SIZE // strides[idx]
        for y in range(cells):
            for x in range(cells):
                anchors.extend([((x + 0.5) / cells, (y + 0.5) / cells)] * repeats)
        idx = last
    return np.array(anchors)


def _weighted_nms(detections: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    """MediaPipe's WEIGHTED non-max suppression.

    All candidates overlapping the current top detection are averaged,
    weighted by score — box and keypoints alike.

    Args:
        detections: Rows of (score, cx, cy, w, h, kp0x, kp0y, ...), normalized.

    Returns:
        Blended detections, same row layout, highest score first.
    """
    remaining = detections[np.argsort(-detections[:, 0])]
    output = []
    while len(remaining):
        top = remaining[0]
        x1 = remaining[:, 1] - remaining[:, 3] / 2
        y1 = remaining[:, 2] - remaining[:, 4] / 2
        x2 = remaining[:, 1] + remaining[:, 3] / 2
        y2 = remaining[:, 2] + remaining[:, 4] / 2
        tx1, ty1, tx2, ty2 = top[1] - top[3] / 2, top[2] - top[4] / 2, top[1] + top[3] / 2, top[2] + top[4] / 2
        inter = np.maximum(0, np.minimum(x2, tx2) - np.maximum(x1, tx1)) * np.maximum(
            0, np.minimum(y2, ty2) - np.maximum(y1, ty1)
        )
        union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - inter
        iou = inter / np.maximum(union, 1e-9)

        overlapping = remaining[iou > iou_threshold]
        weights = overlapping[:, :1]
        blended = top.copy()
        blended[1:] = (overlapping[:, 1:] * weights).sum(axis=0) / weights.sum()
        output.append(blended)
        remaining = remaining[iou <= iou_threshold]
    return np.array(output)


class BlazeFace:
    """BlazeFace short-range ONNX Runtime inference with MediaPipe post-processing."""

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        """Load a BlazeFace short-range ONNX model.

        Args:
            model_path: Path to the ONNX file.
            providers: ONNX Runtime execution providers. Defaults to
                auto-detected (CUDA > CoreML > CPU).
        """
        if providers is None:
            available = ort.get_available_providers()
            preferred = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
            providers = [p for p in preferred if p in available]

        options = ort.SessionOptions()
        options.log_severity_level = 3  # silence CoreML partition warnings
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.anchors = _generate_face_anchors()

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Convert a BGR crop into a CHW float32 slice in [-1, 1], RGB order."""
        rgb = crop[:, :, ::-1].astype(np.float32)
        return np.transpose((rgb - 127.5) / 127.5, (2, 0, 1))

    def detect(self, image: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect faces in a full image.

        Args:
            image: Full BGR image, shape (H, W, 3).
            threshold: Minimum detection confidence.

        Returns:
            A tuple of bboxes (N, 4) as (x1, y1, x2, y2), keypoints (N, 6, 2) —
            rows 0/1 are the viewer-left/right eye — and scores (N,), all
            float32 in full-image pixel coordinates.
        """
        h, w = image.shape[:2]
        scale = FACE_INPUT_SIZE / max(h, w)
        pad_x, pad_y = (FACE_INPUT_SIZE - w * scale) / 2.0, (FACE_INPUT_SIZE - h * scale) / 2.0
        matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]])
        canvas = cv2.warpAffine(image, matrix, (FACE_INPUT_SIZE, FACE_INPUT_SIZE))

        blob = self._preprocess(canvas)[np.newaxis]
        # Outputs in export order: regressors (N, 896, 16) then score logits (N, 896, 1).
        regressors, logits = self.session.run(None, {self.input_name: blob})

        scores = 1.0 / (1.0 + np.exp(-logits[0].ravel().astype(np.float64)))
        keep = scores >= threshold
        if not keep.any():
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, NUM_FACE_KEYPOINTS, 2), dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        reg, anchor = regressors[0][keep].astype(np.float64), self.anchors[keep]
        rows = np.empty((int(keep.sum()), 5 + 2 * NUM_FACE_KEYPOINTS))
        rows[:, 0] = scores[keep]
        rows[:, 1:3] = reg[:, 0:2] / FACE_INPUT_SIZE + anchor  # cx, cy
        rows[:, 3:5] = reg[:, 2:4] / FACE_INPUT_SIZE  # w, h
        for k in range(NUM_FACE_KEYPOINTS):
            rows[:, 5 + 2 * k : 7 + 2 * k] = reg[:, 4 + 2 * k : 6 + 2 * k] / FACE_INPUT_SIZE + anchor
        rows = _weighted_nms(rows)

        def unletterbox(xy_normalized: np.ndarray) -> np.ndarray:
            return (xy_normalized * FACE_INPUT_SIZE - (pad_x, pad_y)) / scale

        centers = unletterbox(rows[:, 1:3])
        halves = rows[:, 3:5] * FACE_INPUT_SIZE / scale / 2.0
        bboxes = np.concatenate([centers - halves, centers + halves], axis=1)
        keypoints = unletterbox(rows[:, 5:].reshape(-1, NUM_FACE_KEYPOINTS, 2))
        return bboxes.astype(np.float32), keypoints.astype(np.float32), rows[:, 0].astype(np.float32)
