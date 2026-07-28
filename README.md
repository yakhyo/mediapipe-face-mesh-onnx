# Face Mesh: 468 dense facial landmarks

[![Downloads](https://img.shields.io/github/downloads/yakhyo/mediapipe-face-mesh-onnx/total.svg?label=Downloads)](https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

> [!TIP]
> The models and functionality in this repository are being integrated into [UniFace](https://github.com/yakhyo/uniface), an all-in-one face analysis toolkit.<br>
> [![PyPI Version](https://img.shields.io/pypi/v/uniface.svg)](https://pypi.org/project/uniface/) [![GitHub Stars](https://img.shields.io/github/stars/yakhyo/uniface)](https://github.com/yakhyo/uniface/stargazers)

[MediaPipe Face Mesh](https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh) turns a 192×192 face crop into 468 dense 3D landmarks, and every face in an image goes through the model in one batched session call. MediaPipe's own BlazeFace short-range detector is bundled here, so there is no separate detector to wire up.

![MediaPipe Face Mesh demo](assets/banner.jpg)

## Install

```bash
pip install -r requirements.txt
```

Then download the weights below into `weights/`.

## Models

| Model | Input | Output | ONNX | PyTorch |
| ----- | ----- | ------ | ---- | ------- |
| Face Mesh (N-batch, default) | Nx3x192x192 | 468 × (x, y, z) | [Link](https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/download/weights/face_mesh_Nx3x192x192.onnx) | [Link](https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/download/weights/face_mesh_192x192.pt) |
| BlazeFace short-range (bundled detector) | Nx3x128x128 | 896 anchors | [Link](https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/download/weights/face_detection_short_range.onnx) | [Link](https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/download/weights/face_detection_short_range.pt) |

Both graphs hold the bare networks. Anchor decoding, weighted NMS, and the mapping from crop coordinates back to the image all run in Python.

Neither ONNX file is a redistribution of somebody else's. `onnx_export.py` builds both from the `.pt` state dicts in this repository, and those trace back through:

```
Google .tflite  ->  intermediate ONNX (read for weight tensors only)  ->  hand-written nn.Module  ->  .pt  ->  onnx_export.py  ->  shipped .onnx
```

- Face Mesh: the intermediate was [PINTO0309's conversion](https://github.com/PINTO0309/facemesh_onnx_tensorrt) of `face_landmark.tflite`
- BlazeFace: a local `tf2onnx` conversion of the `face_detection_short_range.tflite` bundled in the mediapipe 0.10.14 wheel

Float32 throughout, no quantization at any step.

## Run it

```bash
python onnx_inference.py assets/inputs/man_smiling.jpg      # one image
python onnx_inference.py assets/inputs/*.jpg --save-dir assets/results
python onnx_inference.py 0                                  # webcam
python onnx_inference.py 0 --mesh points                    # full | partial | points
```

Same clip through this repo and through original MediaPipe:

| This repo | Original MediaPipe |
| :-: | :-: |
| ![](assets/results/woman_smile.gif) | ![](assets/results/woman_smile_mp.gif) |

### Python

```python
import cv2
from models import BlazeFace, FaceMesh

image = cv2.imread('assets/inputs/man_smiling.jpg')
detector = BlazeFace('weights/face_detection_short_range.onnx')
mesher = FaceMesh('weights/face_mesh_Nx3x192x192.onnx')

bboxes, keypoints, _ = detector.detect(image)
landmarks, scores = mesher.predict(image, bboxes, margin=0.25, keypoints=keypoints)
print(landmarks.shape, scores.shape)  # (N, 468, 3) (N,)
```

`predict` returns two arrays:

| Array | Shape | What it holds |
| ----- | ----- | ------------- |
| `landmarks` | `(N, 468, 3)` | `x` and `y` in full-image pixels, `z` relative depth on the same scale (smaller is closer, comparable only within one face) |
| `scores` | `(N,)` | Face presence probability between 0 and 1 |

The topology is MediaPipe's canonical 468-point mesh, and the drawing edge lists ship in [models/tessellation.py](models/tessellation.py).

> [!IMPORTANT]
> The graph's raw score output is a logit, not a probability. Confident faces land around 20 to 40. `FaceMesh` applies the sigmoid for you, but if you run the ONNX file yourself, don't threshold that raw number as if it were in [0, 1].

## PyTorch

[models/model.py](models/model.py) holds `FaceMeshNet` and `BlazeFaceNet`, hand-written `nn.Module` architectures recovered from MediaPipe's graphs. `inference.py` runs the same detect-then-mesh pipeline on them behind the same CLI:

```bash
python inference.py assets/inputs/woman.jpg --save-dir assets/results
```

- Agrees with the ONNX path to roughly 6e-4 px
- `torch` is wanted only here (`pip install torch`); the ONNX pipeline never imports it
- No BatchNorm layers anywhere, because TFLite folds normalization into the surrounding convolution weights at export time. Google's own `.tflite` files contain no BN ops either.

## Parity with original MediaPipe

Verified against `mediapipe==0.10.14`, the last release whose wheels still carry the legacy `mp.solutions` API and the bundled `face_landmark.tflite`:

| Comparison | Agreement |
| ---------- | --------- |
| Model level, against the original tflite | 1e-4 px |
| Full detect, crop, and mesh pipeline vs `mp.solutions.face_mesh` static mode | ~0.03 px, or 0.01% of inter-ocular distance |

## Notes and limitations

- The crop follows MediaPipe's ROI recipe: a square region built from the long side of the detector box scaled by 1.5 (`--margin 0.25`), rotated so the detector's eye keypoints sit level. The model was trained on exactly that geometry, so tight, stretched, or tilted crops make the mesh worse.
- Each face gets a single pass, matching MediaPipe's static-image flow. MediaPipe's `landmarks_to_roi` step feeds the *next video frame* rather than a second pass over the same image, so it is deliberately not applied here.
- This is the classic 468-point model. It has no iris landmarks (those come from the 478-point attention variant) and no blendshapes.

## License and attribution

[Apache-2.0](LICENSE). The model and the mesh topology originate from Google's [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache-2.0), and the Face Mesh weights were sourced through [PINTO0309's ONNX conversion](https://github.com/PINTO0309/facemesh_onnx_tensorrt) (Apache-2.0). Demo portraits are from [Unsplash](https://unsplash.com) and the demo clip from [Pixabay](https://pixabay.com).
