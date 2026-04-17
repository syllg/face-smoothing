import os
import sys
import logging
import traceback
import cv2


def _resource_path(relative_path):
    # Resolve bundled/resource file path robustly
    normalized = relative_path.replace("\\", "/").lstrip("/")
    frozen = bool(getattr(sys, "frozen", False))

    # Check for PyInstaller _MEIPASS first
    meipass_root = getattr(sys, "_MEIPASS", None)
    if meipass_root:
        # Bundled data at _MEIPASS/face_smoothing/models
        # or _MEIPASS/models
        candidates = [
            os.path.join(meipass_root, "face_smoothing", normalized),
            os.path.join(meipass_root, normalized),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

    if frozen:
        exe_root = os.path.dirname(os.path.abspath(sys.executable))
        candidates = [
            os.path.join(exe_root, "_internal", "face_smoothing", normalized),
            os.path.join(exe_root, "_internal", normalized),
            os.path.join(exe_root, "face_smoothing", normalized),
            os.path.join(exe_root, normalized),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        # Prefer PyInstaller one-dir layout fallback.
        return os.path.join(exe_root, "_internal", "face_smoothing", normalized)

    # Fallback to local source directories
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(package_root)))
    
    candidates = [
        os.path.join(package_root, normalized),
        os.path.join(project_root, normalized),
        os.path.join(project_root, "src", "face_smoothing", normalized),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return os.path.join(package_root, normalized)

class BackendManager:
    """Manages face detection backends and model validation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("net", {}).get("model_name", "buffalo_l")

    def _opencv_cuda_available(self):
        try:
            return bool(hasattr(cv2, "cuda")) and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False

    def log_runtime_status(self):
        opencv_cuda_ok = self._opencv_cuda_available()
        logging.info(f"Runtime OpenCV CUDA available: {opencv_cuda_ok}")
        if opencv_cuda_ok:
            logging.info(
                f"Runtime OpenCV CUDA device count: {cv2.cuda.getCudaEnabledDeviceCount()}"
            )

        try:
            import torch
            torch_cuda = torch.cuda.is_available()
            torch_device = "cuda:0" if torch_cuda else "cpu"
            logging.info(f"Runtime torch.cuda.is_available(): {torch_cuda}")
            logging.info(f"Runtime torch current device: {torch_device}")
        except Exception as error:
            logging.info(f"Runtime torch status unavailable: {error}")

        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            logging.info(f"Runtime ONNX Runtime providers: {providers}")
        except Exception as error:
            logging.info(f"Runtime ONNX Runtime status unavailable: {error}")

    def _required_model_paths(self):
        # We prefer the bundled models in src/face_smoothing/models
        bundled_root = _resource_path("models")
        insightface_root = os.path.join(bundled_root, self.model_name)

        # Development fallback only; frozen builds must be standalone.
        if not os.path.isdir(insightface_root) and not getattr(sys, "frozen", False):
            insightface_root = os.path.join(
                os.path.expanduser("~"), ".insightface", "models", self.model_name
            )

        required = {
            "BiSeNet parsing": _resource_path(
                os.path.join("models", "faceparsing_resnet18.onnx")
            ),
        }
        
        for filename in [
            "det_10g.onnx",
            "2d106det.onnx",
            "1k3d68.onnx",
            "genderage.onnx",
            "w600k_r50.onnx",
        ]:
            required[f"InsightFace {filename}"] = os.path.join(insightface_root, filename)
        return required

    def validate_required_models(self):
        required = self._required_model_paths()
        missing = [
            (name, path) for name, path in required.items() if not os.path.isfile(path)
        ]
        if not missing:
            logging.info("Model check passed: all required model files are available.")
            return

        # If missing in bundled, try to log where we looked
        details = "\n".join([f"- {name}: {path}" for name, path in missing])
        logging.error(f"Model check failed. Missing model files:\n{details}")
        
        raise FileNotFoundError(
            "Model check failed. Ensure 'models' is bundled in the .exe, or available in ~/.insightface/models for development mode."
        )

    def build_detector(self):
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
        except Exception as error:
            raise RuntimeError(
                "InsightFace detector requires 'insightface' and 'onnxruntime-gpu' packages."
            ) from error

        providers = ort.get_available_providers()
        use_cuda = "CUDAExecutionProvider" in providers
        if use_cuda:
            selected_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
            backend_name = "insightface-cuda"
        else:
            selected_providers = ["CPUExecutionProvider"]
            ctx_id = -1
            backend_name = "insightface-cpu"
            logging.warning(
                "CUDAExecutionProvider is unavailable. Falling back to CPUExecutionProvider."
            )

        # Force InsightFace to use our bundled models directory
        # FaceAnalysis(root=X) looks for models in X/models/model_name/
        bundled_models_path = _resource_path("models")
        # Set root to the parent of 'models' folder
        bundled_root = os.path.dirname(bundled_models_path)
        
        logging.info(f"Initializing FaceAnalysis with root: {bundled_root}")
        detector = FaceAnalysis(
            name=self.model_name, 
            root=bundled_root,
            providers=selected_providers,
        )
        
        det_size = tuple(self.cfg.get("net", {}).get("det_size", [640, 640]))
        det_thresh = min(float(self.cfg["net"]["conf_threshold"]), 0.5)
        detector.prepare(
            ctx_id=ctx_id,
            det_size=det_size,
            det_thresh=det_thresh,
        )
        logging.info(
            f"Detector backend selected: {backend_name} (providers={selected_providers}, root: {bundled_root})"
        )
        return detector, backend_name
