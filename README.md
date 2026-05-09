)

# Face Smoothing: Detection and Beautification

***

OpenCV implementation of facial smoothing. Facial detection is done using an pretrained TensorFlow face detection model. Facial smoothing is accomplished using the following steps:

- Change image from BGR to HSV colorspace
- Create mask of HSV image
- Apply a bilateral filter to the Region of Interest
- Apply filtered ROI back to original image

***

## Install

```
git clone https://github.com/syllg/face-filter.git
cd face-filter
pip install -r requirements.txt
```

## Run

### Batch mode

```
python infer.py --input "path/to/input.jpg" --output "data/output"
python infer.py --input "path/to/video.mp4" --output "data/output"
python infer.py --input "path/to/folder" --output "data/output"
```

### Watcher mode (hot-folder)

If `--input` is not provided, the app runs as a folder watcher.

Default folders:

- Input: `~/selfy-time/beauty_input`
- Output: `~/selfy-time/beauty_output`

Example:

```bash
python infer.py --watch-input "C:\Users\yourname\selfy-time\beauty_input" --watch-output "C:\Users\yourname\selfy-time\beauty_output" --parallel-workers 3
```

Useful flags:

- `--parallel-workers`: number of worker threads
- `--processing-timeout-seconds`: timeout for stuck processing detection
- `--stuck-check-interval-seconds`: interval for stuck checker
- `--processed-ttl-seconds`: retention time for processed tracker entries
- `--save-steps`: save concatenated processing steps image
- `--show-detections`: output image with bounding boxes

#### Example: --save-steps flag

!\[alt text]\(https\://github.com/syllg/face-filter/blob/main/data/output/combined\_0.jpg?raw=true Processing steps)

## Build Windows executable (.exe)

This project includes an auto-py-to-exe config:

- [settings\_face\_smoothing.json](file:///c:/Users/sylvi/works/face-smoothing/settings_face_smoothing.json)

Steps:

1. Install build tools

```bash
pip install auto-py-to-exe pyinstaller
```

1. Run UI

```bash
auto-py-to-exe
```

1. Import config JSON: `settings_face_smoothing.json`
2. Build

If you changed packaging settings before, remove old artifacts first:

```bash
rmdir /s /q build
rmdir /s /q dist
rmdir /s /q output
```

Quick test:

```bash
output\infer\infer.exe --help
```

If startup fails, check:

- `C:\Users\<username>\face-smoothing.log`

## Reference
https://github.com/5starkarma/face-smoothing

## TODO

- [x] Finish documentation and cleanup functions
- [x] Reduce input image size for detections
- [x] Fix combined output
- [x] Test on multiple faces
- [x] Apply blurring on multiple faces
- [x] Video inference
- [x] Save bounding box to output
- [x] **New: Hot-folder watcher mode for automated processing**
- [x] **New: Parallel processing with multi-worker threads**
- [x] **New: Intelligent processing tracker (deduplication & state tracking)**
- [x] **New: Automatic recovery for "stuck" files**
- [x] **New: Configurable TTL for processed file history**
- [x] **New: Portable Windows executable (.exe) support**
- [x] **New: Persistent logging and crash reporting**
- [ ] Apply different blurring techniques/advanced algo using facial landmarks to blur only skin regions
- [ ] Unit tests
- [ ] Run time tests on units

