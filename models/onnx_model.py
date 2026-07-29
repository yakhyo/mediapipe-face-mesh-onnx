# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""MediaPipe Face Mesh ONNX Runtime inference.

The model (`face_mesh_Nx3x192x192.onnx`) is the pure 468-point Face Mesh network
exported from this repo's PyTorch `FaceMeshNet` (see onnx_export.py). It takes a
batch of 192x192 RGB face crops in [0, 1] and returns raw landmarks in 192-crop
pixels plus a face-presence logit; all post-processing is done here in Python.
(The old PINTO graph instead baked a crop-offset remap and a final int32 cast
into the ONNX; standardizing on the PyTorch export removes both.)

The crop fed to the model follows MediaPipe's ROI recipe — its `detection_to_roi`
rule: a SQUARE region (long side of the detector box, scaled 1.5x, never a
stretched rectangle) rotated so the eye line is horizontal (angle from the
detector's eye keypoints). The model runs once per face — MediaPipe's
static-image flow (`mp.solutions.face_mesh`, `static_image_mode=True`): detect,
crop, mesh. Verified against MediaPipe's graph configs: identical 1.5x /
square_long scale and eye-keypoint rotation, single pass (no landmarks_to_roi
tracking, which MediaPipe uses only across video frames).

Landmarks come back in 192-crop pixels and are mapped to full-image coordinates
by the inverse of the ROI warp (`warp_roi`); z is scaled onto the same pixel
scale. The score output is a raw logit (confident faces score 20-40), so
`sigmoid` is applied before it is exposed.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

# Session and preprocessing are kept deliberately minimal here — UniFace's base
# classes take over those concerns when these models are integrated there.

#: Defaults for the 468-point Face Mesh. `FaceMesh` reads the actual values from the
#: graph it loads, so the 478-point Face Landmarker (256x256) needs no change here.
NUM_LANDMARKS = 468
INPUT_SIZE = (192, 192)

# An ROI is (center_x, center_y, side, angle_degrees) in image coordinates.
Roi = tuple[float, float, float, float]


def roi_from_box(bbox: np.ndarray, keypoints: np.ndarray | None = None, margin: float = 0.25) -> Roi:
    """Build the ROI from a detector box, MediaPipe's detection_to_roi rule.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2).
        keypoints: Optional 5-point landmarks, shape (5, 2) — rows 0 and 1 are
            the left and right eye; they set the ROI rotation (roll).
        margin: Fraction of box size to expand by on each side; the default
            0.25 yields MediaPipe's 1.5x scale.

    Returns:
        A square, optionally rotated ROI as (center_x, center_y, side, angle).
    """
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    side = (1.0 + 2.0 * margin) * max(x2 - x1, y2 - y1)

    angle = 0.0
    if keypoints is not None:
        dx, dy = (float(v) for v in np.asarray(keypoints)[1] - np.asarray(keypoints)[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))

    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, side, angle


def warp_roi(image: np.ndarray, roi: Roi, size: int = INPUT_SIZE[0]) -> tuple[np.ndarray, np.ndarray]:
    """Sample a rotated square ROI directly at the model's input resolution.

    Single bilinear resampling straight to the model size — mirroring
    MediaPipe's ImageToTensorCalculator, which never materializes an
    intermediate crop (a warp-then-resize implementation interpolates twice
    and quantizes the crop side to whole pixels). Out-of-image regions are
    zero-padded, like MediaPipe's own cropper.

    Args:
        image: Full BGR image, shape (H, W, 3).
        roi: The (center_x, center_y, side, angle) ROI to cut.
        size: Output edge length in pixels; the model's input resolution.

    Returns:
        A tuple of the (size, size, 3) crop and the inverse 2x3 affine mapping
        crop-pixel coordinates back to full-image coordinates.
    """
    cx, cy, side, angle = roi
    out = size
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, out / side)
    matrix[0, 2] += out / 2.0 - cx
    matrix[1, 2] += out / 2.0 - cy

    crop = cv2.warpAffine(image, matrix, (out, out))
    return crop, cv2.invertAffineTransform(matrix)


class FaceMesh:
    """MediaPipe Face Mesh (468 landmarks) ONNX Runtime inference.

    All faces of an image are batched into a single session run — the graph has
    a dynamic batch dimension. Results are plain numpy arrays; typed result
    containers arrive with the UniFace integration.

    Source model: exported from this repo's PyTorch `FaceMeshNet` (models/model.py),
    an architecture recovered from MediaPipe's Face Mesh
    (https://github.com/google-ai-edge/mediapipe).
    """

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        """Load a Face Mesh ONNX model.

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

        # Read the geometry from the graph rather than hardcoding it, so the same class
        # serves Face Mesh (468 @ 192) and Face Landmarker (478 @ 256).
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_size = int(input_meta.shape[2])
        self.num_landmarks = int(self.session.get_outputs()[0].shape[1])

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Convert a BGR crop into a CHW float32 slice in [0, 1], RGB order."""
        if crop.shape[:2] != (self.input_size, self.input_size):
            crop = cv2.resize(crop, (self.input_size, self.input_size))
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))

    def _run(self, blobs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Run the pure network on a batch of 192x192 blobs.

        Returns landmarks (N, 468, 3) in 192-crop pixels and presence scores
        (N,) in [0, 1] — the graph's score output is a raw logit, sigmoided here.
        """
        landmarks, logits = self.session.run(None, {self.input_name: np.stack(blobs)})
        scores = 1.0 / (1.0 + np.exp(-logits.ravel().astype(np.float64)))
        return landmarks.astype(np.float32), scores.astype(np.float32)

    def _run_rois(self, image: np.ndarray, rois: list[Roi]) -> tuple[np.ndarray, np.ndarray]:
        blobs, inverses = [], []
        for roi in rois:
            crop, inverse = warp_roi(image, roi, self.input_size)
            blobs.append(self._preprocess(crop))
            inverses.append(inverse)

        landmarks, scores = self._run(blobs)
        landmarks = landmarks.astype(np.float64)
        for idx, (roi, inverse) in enumerate(zip(rois, inverses, strict=True)):
            landmarks[idx, :, :2] = landmarks[idx, :, :2] @ inverse[:, :2].T + inverse[:, 2]
            landmarks[idx, :, 2] *= roi[2] / self.input_size  # z onto the same image-pixel scale
        return landmarks.astype(np.float32), scores

    def predict(
        self,
        image: np.ndarray,
        bboxes: list[np.ndarray] | np.ndarray,
        margin: float = 0.25,
        keypoints: list[np.ndarray] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict dense landmarks for every detected face of a full image.

        Single pass per face — MediaPipe's static-image flow.

        Args:
            image: Full BGR image, shape (H, W, 3).
            bboxes: One (x1, y1, x2, y2) box per face, e.g. from a detector.
            margin: Fraction of box size to expand each crop by on each side.
            keypoints: Optional 5-point landmarks per face, shape (5, 2) each,
                aligned with `bboxes`. When given, the ROI is roll-normalized
                around the eye line, keeping the mesh stable on tilted heads.

        Returns:
            A tuple of landmarks with shape (N, 468, 3), float32 — `x`/`y` in
            full-image pixels, `z` relative depth on the same scale (smaller is
            closer) — and face-presence scores with shape (N,) in [0, 1].
        """
        if len(bboxes) == 0:
            return np.empty((0, self.num_landmarks, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
        if keypoints is not None and len(keypoints) != len(bboxes):
            raise ValueError(f'Got {len(bboxes)} boxes but {len(keypoints)} keypoint sets')

        rois = [
            roi_from_box(bbox, None if keypoints is None else keypoints[idx], margin) for idx, bbox in enumerate(bboxes)
        ]
        landmarks, scores = self._run_rois(image, rois)
        return landmarks, scores
