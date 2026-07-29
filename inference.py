# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""PyTorch inference for Face Mesh — runs the reference nets directly.

The counterpart to `onnx_inference.py`: instead of the ONNX models it runs the
PyTorch modules from `models/model.py` — `BlazeFaceNet` for detection and, for the
landmarks, either `FaceMeshNet` (468 points, 192x192) or `FaceLandmarkerNet` (478
points with irises, 256x256). The pure post-processing (anchor decode, weighted NMS,
ROI recipe, warp, inverse-affine) is reused from the same helpers the ONNX path uses,
so the pipeline is identical apart from the model backend.

Both landmark models share one ROI recipe, verified against MediaPipe's own output:
the 256x256 model needs no recalibration of `--margin`.

Needs torch (`pip install torch`).

    python inference.py assets/inputs/woman.jpg --save-dir assets/results
    python inference.py 0                                # webcam
    python inference.py 0 --model landmarker             # 478 points, with irises
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import torch

from models.blazeface import (
    FACE_INPUT_SIZE,
    NUM_FACE_KEYPOINTS,
    _generate_face_anchors,
    _weighted_nms,
)
from models.model import BlazeFaceNet, FaceLandmarkerNet, FaceMeshNet
from models.onnx_model import roi_from_box, warp_roi
from models.tessellation import (
    FACEMESH_TESSELATION_FULL,
    FACEMESH_TESSELATION_PARTIAL,
    IRIS_LEFT,
    IRIS_RIGHT,
    NUM_MESH_LANDMARKS,
)

MESH_COLOR, POINT_COLOR = (230, 230, 60), (0, 209, 255)  # BGR (cyan mesh, amber points)
IRIS_COLOR = (60, 60, 255)  # BGR red, so the iris points stand out from the mesh
FACE_ANCHORS = _generate_face_anchors()

# name -> (net class, checkpoint, input resolution)
LANDMARK_MODELS = {
    'mesh': (FaceMeshNet, 'weights/face_mesh_192x192.pt', 192),
    'landmarker': (FaceLandmarkerNet, 'weights/face_landmarker_256x256.pt', 256),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MediaPipe Face Mesh PyTorch inference - 468 dense landmarks for images or a webcam stream',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'sources',
        type=str,
        nargs='+',
        help='Image paths and/or webcam indices (e.g. 0)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='mesh',
        choices=list(LANDMARK_MODELS),
        help='mesh = 468 points @192, landmarker = 478 points with irises @256',
    )
    parser.add_argument('--mesh-weights', type=str, default=None, help='Override the landmark model .pt')
    parser.add_argument(
        '--detector-weights',
        type=str,
        default='weights/face_detection_short_range.pt',
        help='BlazeFaceNet .pt',
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=0.25,
        help='Fraction of box size to expand each crop by',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Face-presence probability below which to discard',
    )
    parser.add_argument(
        '--mesh',
        type=str,
        default='full',
        choices=['full', 'partial', 'points'],
        help='Drawing style',
    )
    parser.add_argument('--save-dir', type=str, default=None, help='If set, save annotated images here')
    return parser.parse_args()


def load_net(net_cls: type, weights: str) -> torch.nn.Module:
    net = net_cls().eval()
    net.load_state_dict(torch.load(weights))
    return net


@torch.no_grad()
def detect_faces(net: BlazeFaceNet, image: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """BlazeFace detection with the PyTorch net. Returns (bboxes (N, 4), keypoints (N, 6, 2)) in image pixels."""
    h, w = image.shape[:2]
    scale = FACE_INPUT_SIZE / max(h, w)
    pad_x, pad_y = (
        (FACE_INPUT_SIZE - w * scale) / 2.0,
        (FACE_INPUT_SIZE - h * scale) / 2.0,
    )
    matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]])
    canvas = cv2.warpAffine(image, matrix, (FACE_INPUT_SIZE, FACE_INPUT_SIZE))
    blob = np.transpose((canvas[:, :, ::-1].astype(np.float32) - 127.5) / 127.5, (2, 0, 1))[np.newaxis]

    regressors, logits = (t.numpy() for t in net(torch.from_numpy(blob)))
    scores = 1.0 / (1.0 + np.exp(-logits[0].ravel().astype(np.float64)))
    keep = scores >= threshold
    if not keep.any():
        return np.empty((0, 4), np.float32), np.empty((0, NUM_FACE_KEYPOINTS, 2), np.float32)

    reg, anchor = regressors[0][keep].astype(np.float64), FACE_ANCHORS[keep]
    rows = np.empty((int(keep.sum()), 5 + 2 * NUM_FACE_KEYPOINTS))
    rows[:, 0] = scores[keep]
    rows[:, 1:3] = reg[:, 0:2] / FACE_INPUT_SIZE + anchor  # cx, cy
    rows[:, 3:5] = reg[:, 2:4] / FACE_INPUT_SIZE  # w, h
    for k in range(NUM_FACE_KEYPOINTS):
        rows[:, 5 + 2 * k : 7 + 2 * k] = reg[:, 4 + 2 * k : 6 + 2 * k] / FACE_INPUT_SIZE + anchor
    rows = _weighted_nms(rows)

    def unletterbox(xy: np.ndarray) -> np.ndarray:
        return (xy * FACE_INPUT_SIZE - (pad_x, pad_y)) / scale

    centers = unletterbox(rows[:, 1:3])
    halves = rows[:, 3:5] * FACE_INPUT_SIZE / scale / 2.0
    bboxes = np.concatenate([centers - halves, centers + halves], axis=1)
    keypoints = unletterbox(rows[:, 5:].reshape(-1, NUM_FACE_KEYPOINTS, 2))
    return bboxes.astype(np.float32), keypoints.astype(np.float32)


@torch.no_grad()
def predict_mesh(
    net: torch.nn.Module,
    image: np.ndarray,
    bboxes: np.ndarray,
    keypoints: np.ndarray,
    margin: float = 0.25,
    input_size: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense landmarks with the PyTorch net. Returns (landmarks (N, K, 3), scores (N,)).

    K is 468 for `FaceMeshNet` and 478 for `FaceLandmarkerNet`; `input_size` must match
    whichever net was passed.
    """
    num_landmarks = getattr(net, 'NUM_LANDMARKS', 468)
    if len(bboxes) == 0:
        return np.empty((0, num_landmarks, 3), np.float32), np.empty(0, np.float32)

    rois = [roi_from_box(bbox, keypoints[idx], margin) for idx, bbox in enumerate(bboxes)]
    blobs, inverses = [], []
    for roi in rois:
        crop, inverse = warp_roi(image, roi, input_size)
        blobs.append(np.transpose(crop[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1)))
        inverses.append(inverse)

    landmarks, logits = (t.numpy() for t in net(torch.from_numpy(np.stack(blobs))))
    scores = 1.0 / (1.0 + np.exp(-logits.ravel().astype(np.float64)))
    landmarks = landmarks.astype(np.float64)
    for idx, (roi, inverse) in enumerate(zip(rois, inverses, strict=True)):
        landmarks[idx, :, :2] = landmarks[idx, :, :2] @ inverse[:, :2].T + inverse[:, 2]
        landmarks[idx, :, 2] *= roi[2] / input_size
    return landmarks.astype(np.float32), scores.astype(np.float32)


def annotate(image: np.ndarray, landmarks: np.ndarray, mesh_mode: str) -> None:
    """Draw one face's mesh (tessellation lines and/or landmark points) onto `image`.

    Accepts 468 or 478 points. The tessellation indexes only the first 468, which the
    478-point model keeps in the same order; the irises are drawn separately via
    IRIS_LEFT / IRIS_RIGHT, with a circle fitted to each.
    """
    points = np.rint(landmarks[:, :2]).astype(np.int32)
    mesh = points[:NUM_MESH_LANDMARKS]

    if mesh_mode in ('full', 'partial'):
        edges = FACEMESH_TESSELATION_FULL if mesh_mode == 'full' else FACEMESH_TESSELATION_PARTIAL
        for start_idx, end_idx in edges:
            cv2.line(image, mesh[start_idx], mesh[end_idx], MESH_COLOR, 1, cv2.LINE_AA)

    face_size = mesh[:, 1].max() - mesh[:, 1].min()
    radius = max(1, round(face_size / 120))
    if mesh_mode in ('partial', 'points'):
        for x, y in mesh:
            cv2.circle(image, (x, y), radius, POINT_COLOR, -1, cv2.LINE_AA)

    # Circle each iris and mark its five points. Empty on a 468-point result, so this
    # is a no-op there rather than a branch.
    for iris in (points[IRIS_LEFT], points[IRIS_RIGHT]):
        if len(iris) == 0:
            continue
        center = iris[0]
        iris_radius = round(float(np.linalg.norm(iris[1:] - center, axis=1).mean()))
        cv2.circle(image, center, max(iris_radius, 1), IRIS_COLOR, 1, cv2.LINE_AA)
        for x, y in iris:
            cv2.circle(image, (x, y), max(radius, 2), IRIS_COLOR, -1, cv2.LINE_AA)


def run_image(
    image_path: str,
    mesh_net: torch.nn.Module,
    det_net: BlazeFaceNet,
    args: argparse.Namespace,
    input_size: int,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f'{image_path}: READ_ERR')
        return

    bboxes, keypoints = detect_faces(det_net, image, args.threshold)
    if len(bboxes) == 0:
        print(f'{image_path}: NO_FACE')
        return
    landmarks, scores = predict_mesh(mesh_net, image, bboxes, keypoints, args.margin, input_size)

    for idx, (face_landmarks, score) in enumerate(zip(landmarks, scores, strict=True), start=1):
        kept = score >= args.threshold
        print(f'{image_path} [face {idx}]: score={score:.4f}{"" if kept else " -> below threshold, skipped"}')
        if kept and args.save_dir:
            annotate(image, face_landmarks, args.mesh)

    if args.save_dir:
        output_path = os.path.join(args.save_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        print(f'Saved {output_path}')


def run_webcam(
    camera_index: int,
    mesh_net: torch.nn.Module,
    det_net: BlazeFaceNet,
    args: argparse.Namespace,
    input_size: int,
) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f'webcam {camera_index}: OPEN_ERR')
        return

    window = f'Face Mesh (PyTorch) - webcam {camera_index}'
    controls = 'q/ESC to quit' + (', s to save a snapshot' if args.save_dir else '')
    print(f'webcam {camera_index}: {controls}')

    while True:
        ok, frame = capture.read()
        if not ok:
            print(f'webcam {camera_index}: READ_ERR')
            break

        bboxes, keypoints = detect_faces(det_net, frame, args.threshold)
        if len(bboxes) > 0:
            landmarks, scores = predict_mesh(mesh_net, frame, bboxes, keypoints, args.margin, input_size)
            for face_landmarks, score in zip(landmarks, scores, strict=True):
                if score >= args.threshold:
                    annotate(frame, face_landmarks, args.mesh)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('s') and args.save_dir:
            output_path = os.path.join(
                args.save_dir,
                f'webcam{camera_index}_{time.strftime("%Y%m%d-%H%M%S")}.jpg',
            )
            cv2.imwrite(output_path, frame)
            print(f'Saved {output_path}')

    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    net_cls, default_weights, input_size = LANDMARK_MODELS[args.model]
    mesh_net = load_net(net_cls, args.mesh_weights or default_weights)
    det_net = load_net(BlazeFaceNet, args.detector_weights)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for source in args.sources:
        if source.isdigit():
            run_webcam(int(source), mesh_net, det_net, args, input_size)
        else:
            run_image(source, mesh_net, det_net, args, input_size)


if __name__ == '__main__':
    main()
