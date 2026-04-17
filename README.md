# Face Smoothing

Face smoothing pipeline for images/videos with:
- InsightFace face detection + landmarks
- BiSeNet face parsing
- Skin-mask smoothing + exclusion masks
- Batch mode and hot-folder watcher mode
- Windows standalone `.exe` build via PyInstaller

## Project Layout

```text
src/
  face_smoothing/
    configs/
      configs.yaml
    configs/
    detector/
      backend.py
      detect.py
      smooth.py
    services/
      face_smoothing_service.py
      tracker.py
      watcher.py
    utils/
      config.py
      image.py
      video.py
      shared.py
      types.py
    main.py
    models/
      buffalo_l/
        1k3d68.onnx
        2d106det.onnx
        det_10g.onnx
        genderage.onnx
        w600k_r50.onnx
      faceparsing_resnet18.onnx
      face_detection_yunet_2023mar.onnx
      opencv_face_detector.pb
      opencv_face_detector.pbtxt
infer.py
infer.spec
```

## Install

```bash
pip install -r requirements.txt
```

## Run (Development)

### Batch mode

```bash
python infer.py --input "path/to/input.jpg" --output "data/output"
python infer.py --input "path/to/folder" --output "data/output"
python infer.py --input "path/to/video.mp4" --output "data/output"
```

### Watcher mode

If `--input` is not provided, app runs in watcher mode.

Default folders:
- Input: `~/selfy-time/beauty_input`
- Output: `~/selfy-time/beauty_output`

Example:

```powershell
python infer.py --watch-input "C:\Users\yourname\selfy-time\beauty_input" --watch-output "C:\Users\yourname\selfy-time\beauty_output" --parallel-workers 3
```

Useful flags:
- `--input`: path ke file gambar, video, atau folder berisi gambar.
- `--output`: folder tujuan hasil output. Default: `data/output`.
- `--parallel`: jumlah worker paralel di mode batch (dalam satu proses).
- `--save-detection`: simpan image dengan bounding box deteksi wajah.
- `--save-parsing`: simpan mask parsing dan area smoothing (debug).
- `--save-steps`: gabungkan dan simpan semua step pipeline ke satu gambar.
- `--show-detections`: gunakan output dengan bounding box di hasil akhir.
- `--check-models-only`: hanya cek kelengkapan model dan keluar (tanpa proses gambar).

Watcher specific:
- `--watch-input`: override folder input watcher (default `~/selfy-time/beauty_input`).
- `--watch-output`: override folder output watcher (default `~/selfy-time/beauty_output`).
- `--parallel-workers`: jumlah worker thread di mode watcher.

## Models Required

Semua model di-bundle di `src/face_smoothing/models/` dan ikut dimasukkan ke `.exe`:

- `faceparsing_resnet18.onnx`  
  BiSeNet face parsing, menghasilkan peta kelas wajah (kulit, rambut, dll.).
- `buffalo_l/1k3d68.onnx`  
  3D landmark model (68 titik) untuk alignment/detail.
- `buffalo_l/2d106det.onnx`  
  2D landmark model (106 titik) untuk parsing lebih presisi.
- `buffalo_l/det_10g.onnx`  
  Face detection model utama (bounding box).
- `buffalo_l/genderage.onnx`  
  Model gender/age (opsional, tidak kritikal untuk smoothing).
- `buffalo_l/w600k_r50.onnx`  
  Face recognition backbone (dipakai InsightFace, tidak langsung dipakai smoothing).
- `face_detection_yunet_2023mar.onnx`  
  Alternatif detector (tidak aktif default, tapi ter-bundle sebagai opsi).
- `opencv_face_detector.pb` / `opencv_face_detector.pbtxt`  
  Model dan config detector OpenCV (legacy/opsional).

## Build Windows `.exe` (Standalone)

Use the provided spec file:

```powershell
python -m PyInstaller --noconfirm --clean infer.spec
```

Output:

```text
dist/infer/infer.exe
```

Quick validation:

```powershell
.\dist\infer\infer.exe --help
.\dist\infer\infer.exe --check-models-only
```

## Runtime Notes

- Frozen mode resolves bundled assets from PyInstaller bundle paths (`_internal/face_smoothing/...`).
- Config and model assets are expected to be bundled from:
  - `src/face_smoothing/configs`
  - `src/face_smoothing/models`
- Startup/runtime logs are written to:
  - `C:\Users\<username>\face-smoothing.log`

## Troubleshooting

- `configs.yaml not found`:
  rebuild with `--clean` and run the latest `dist/infer/infer.exe`.
- New file in watcher not processed:
  watcher waits for file stabilization and tracker re-queues modified files based on mtime.
