# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import argparse
import os
import time

import cv2
import numpy as np

from models import (
    FACEMESH_TESSELATION_FULL,
    FACEMESH_TESSELATION_PARTIAL,
    BlazeFace,
    FaceMesh,
)
from models.tessellation import IRIS_LEFT, IRIS_RIGHT, NUM_MESH_LANDMARKS

MESH_COLOR, POINT_COLOR = (230, 230, 60), (0, 209, 255)  # BGR (cyan mesh, amber points)
IRIS_COLOR = (60, 60, 255)  # BGR red, so the iris points stand out from the mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MediaPipe Face Mesh ONNX inference - 468 dense landmarks for images or a webcam stream',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'sources',
        type=str,
        nargs='+',
        help='Image paths and/or webcam indices (e.g. 0 opens the default camera)',
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='weights/face_mesh_Nx3x192x192.onnx',
        help='Path to the Face Mesh ONNX model',
    )
    parser.add_argument(
        '--blazeface-model',
        type=str,
        default='weights/face_detection_short_range.onnx',
        help='Path to the BlazeFace short-range detector ONNX model',
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=0.25,
        help='Fraction of box size to expand each detected face crop by on each side',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Face-presence probability below which a mesh is discarded',
    )
    parser.add_argument(
        '--mesh',
        type=str,
        default='full',
        choices=['full', 'partial', 'points'],
        help='Drawing style: full/partial tessellation lines, or landmark points only',
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='If set, save annotated images to this directory',
    )
    return parser.parse_args()


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


def run_image(image_path: str, mesher: FaceMesh, detector: BlazeFace, args: argparse.Namespace) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f'{image_path}: READ_ERR')
        return

    bboxes, keypoints, _ = detector.detect(image)
    if len(bboxes) == 0:
        print(f'{image_path}: NO_FACE')
        return
    landmarks, scores = mesher.predict(
        image,
        bboxes,
        margin=args.margin,
        keypoints=keypoints,
    )

    for idx, (face_landmarks, score) in enumerate(zip(landmarks, scores, strict=True), start=1):
        kept = score >= args.threshold
        print(f'{image_path} [face {idx}]: score={score:.4f}{"" if kept else " -> below threshold, skipped"}')
        if kept and args.save_dir:
            annotate(image, face_landmarks, args.mesh)

    if args.save_dir:
        output_path = os.path.join(args.save_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        print(f'Saved {output_path}')


def run_webcam(camera_index: int, mesher: FaceMesh, detector: BlazeFace, args: argparse.Namespace) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f'webcam {camera_index}: OPEN_ERR')
        return

    window = f'Face Mesh - webcam {camera_index}'
    controls = 'q/ESC to quit' + (', s to save a snapshot' if args.save_dir else '')
    print(f'webcam {camera_index}: {controls}')

    while True:
        ok, frame = capture.read()
        if not ok:
            print(f'webcam {camera_index}: READ_ERR')
            break

        bboxes, keypoints, _ = detector.detect(frame)
        if len(bboxes) > 0:
            landmarks, scores = mesher.predict(
                frame,
                bboxes,
                margin=args.margin,
                keypoints=keypoints,
            )
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

    mesher = FaceMesh(args.model)
    detector = BlazeFace(args.blazeface_model)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for source in args.sources:
        if source.isdigit():
            run_webcam(int(source), mesher, detector, args)
        else:
            run_image(source, mesher, detector, args)


if __name__ == '__main__':
    main()
