import os
import sys
import logging
import traceback
import time
import argparse
import gc
from queue import Queue, Empty
from threading import Thread, current_thread
from dotenv import load_dotenv

# Infrastructure
LOG_PATH = os.path.join(os.path.expanduser("~"), "face-smoothing.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, encoding="utf-8")],
)

# Robust path handling for PyInstaller and normal execution
if getattr(sys, "frozen", False):
    # Running in PyInstaller bundle
    bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    internal_dir = os.path.join(bundle_dir, "_internal")
    
    # Ensure _internal is in path for relative imports if needed
    for p in [bundle_dir, internal_dir]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
else:
    # Running in normal python environment
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def _configure_cuda_dll_paths():
    if os.name != "nt":
        return
    try:
        import site
        user_site = site.getusersitepackages()
        nvidia_base = os.path.join(user_site, "nvidia")
        candidate_dirs = [
            os.path.join(nvidia_base, "cudnn", "bin"),
            os.path.join(nvidia_base, "cublas", "bin"),
            os.path.join(nvidia_base, "cuda_nvrtc", "bin"),
        ]
        existing_path = os.environ.get("PATH", "")
        path_chunks = existing_path.split(os.pathsep) if existing_path else []
        for dll_dir in candidate_dirs:
            if os.path.isdir(dll_dir):
                try:
                    os.add_dll_directory(dll_dir)
                except Exception as error:
                    logging.warning(f"Failed to add DLL directory: {dll_dir} ({error})")
                if dll_dir not in path_chunks:
                    path_chunks.insert(0, dll_dir)
        os.environ["PATH"] = os.pathsep.join(path_chunks)
    except Exception as error:
        logging.warning(f"CUDA DLL path configuration failed: {error}")

_configure_cuda_dll_paths()
load_dotenv()

# Project imports
from face_smoothing.detector.backend import BackendManager
from face_smoothing.utils.config import load_configs, _positive_int, _get_env_int, _get_env_str
from face_smoothing.services.tracker import ProcessingTracker
from face_smoothing.services.face_smoothing_service import FaceSmoothingService
from face_smoothing.services.watcher import HotFolderWatcher, WatcherHandler, scan_existing_files, periodic_stuck_checker

# Constants
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
DEFAULT_PARALLEL_WORKERS = 3
DEFAULT_STUCK_CHECK_INTERVAL = 1800
DEFAULT_PROCESSING_TIMEOUT = 1800
DEFAULT_PROCESSED_TTL_SECONDS = 86400

def _process_worker(queue, tracker, smoothing_service):
    """Background worker that pulls files from the queue and processes them."""
    worker_name = current_thread().name
    while True:
        try:
            filepath = queue.get(timeout=2)
        except Empty:
            continue

        attempt_id = tracker.claim_for_processing(filepath, worker_name)
        if attempt_id is None:
            queue.task_done()
            continue

        success = smoothing_service.process_file(filepath, attempt_id, tracker, worker_name=worker_name)
        tracker.finish_processing(filepath, attempt_id, success, worker_name)
        
        queue.task_done()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Face Smoothing: Detection and Beautification")
    parser.add_argument("--input", type=str, help="Input file path or folder")
    parser.add_argument("--output", type=str, help="Output folder")
    parser.add_argument("--save-detection", action="store_true", help="Save detection results")
    parser.add_argument("--save-parsing", action="store_true", help="Save parsing results")
    parser.add_argument("--save-steps", action="store_true", help="Save intermediate processing steps")
    parser.add_argument("--show-detections", action="store_true", help="Show bounding boxes in final output")
    parser.add_argument("--parallel", type=_positive_int, help="Number of parallel workers")
    parser.add_argument("--check-models-only", action="store_true", help="Check for required models and exit")

    # Watcher-specific args
    parser.add_argument("--watch-input", type=str, help="Input folder for watcher")
    parser.add_argument("--watch-output", type=str, help="Output folder for watcher")
    parser.add_argument("--parallel-workers", type=_positive_int, help="Number of workers for watcher")

    args = parser.parse_args()

    # Configuration loading
    cfg = load_configs()
    backend = BackendManager(cfg)
    
    if args.check_models_only:
        backend.validate_required_models()
        logging.info("Model check complete. Exiting without running inference.")
        return

    backend.log_runtime_status()
    backend.validate_required_models()
    detector, backend_name = backend.build_detector()

    # Initialization of services
    smoothing_service = FaceSmoothingService(cfg, detector, args)
    processing_queue = Queue()
    tracker = ProcessingTracker(processing_queue)

    # Determine mode: Watcher or Batch
    watch_input = args.watch_input or _get_env_str("WATCH_INPUT", None)
    
    if watch_input or not args.input:
        # WATCHER MODE
        input_folder = watch_input or os.path.join(os.path.expanduser("~"), "selfy-time", "beauty_input")
        output_folder = args.watch_output or _get_env_str("WATCH_OUTPUT", os.path.join(os.path.expanduser("~"), "selfy-time", "beauty_output"))
        
        # Override args.output for watcher mode
        args.output = output_folder
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        num_workers = args.parallel_workers or _get_env_int("PARALLEL_WORKERS", DEFAULT_PARALLEL_WORKERS)
        stuck_interval = _get_env_int("STUCK_CHECK_INTERVAL_SECONDS", DEFAULT_STUCK_CHECK_INTERVAL)
        timeout = _get_env_int("PROCESSING_TIMEOUT_SECONDS", DEFAULT_PROCESSING_TIMEOUT)
        ttl = _get_env_int("PROCESSED_TTL_SECONDS", DEFAULT_PROCESSED_TTL_SECONDS)

        logging.info(f"Mode File Watcher (Parallel) - Input: {input_folder}")
        logging.info(f"Workers: {num_workers}, Stuck Check: {stuck_interval}s, TTL: {ttl}s")

        # Start background threads
        for i in range(num_workers):
            t = Thread(target=_process_worker, args=(processing_queue, tracker, smoothing_service), name=f"worker-{i}", daemon=True)
            t.start()

        Thread(target=periodic_stuck_checker, args=(tracker, stuck_interval, timeout, ttl), name="stuck-checker", daemon=True).start()

        # Initial scan
        existing = scan_existing_files(input_folder, VALID_EXTENSIONS)
        if existing:
            logging.info(f"Ditemukan {len(existing)} file awal, queuing...")
            for f in existing:
                tracker.enqueue_if_needed(f, source="initial-scan", valid_extensions=VALID_EXTENSIONS)

        # Start Watcher
        handler = WatcherHandler(tracker, VALID_EXTENSIONS)
        observer = HotFolderWatcher(tracker, VALID_EXTENSIONS)
        observer.schedule(handler, input_folder, recursive=False)
        observer.start()

        logging.info("Tekan Ctrl+C untuk menghentikan")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Berhenti...")
            observer.stop()
            observer.join(timeout=5)
    else:
        # BATCH MODE
        if not args.output:
            args.output = "data/output"
        os.makedirs(args.output, exist_ok=True)

        input_path = os.path.abspath(args.input)
        files_to_process = []
        if os.path.isdir(input_path):
            files_to_process = scan_existing_files(input_path, VALID_EXTENSIONS)
        elif os.path.isfile(input_path):
            files_to_process = [input_path]

        if not files_to_process:
            logging.error(f"Tidak ada file valid ditemukan di: {input_path}")
            return

        logging.info(f"Mode batch processing: {len(files_to_process)} file(s) ditemukan")
        
        num_parallel = args.parallel or 1
        if num_parallel > 1:
            from concurrent.futures import ThreadPoolExecutor
            # In batch mode, we start workers only once, then wait for the queue to be empty
            with ThreadPoolExecutor(max_workers=num_parallel) as executor:
                for _ in range(num_parallel):
                    executor.submit(_process_worker, processing_queue, tracker, smoothing_service)
                
                for f in files_to_process:
                    tracker.enqueue_if_needed(f, source="batch-mode", valid_extensions=VALID_EXTENSIONS)
            
            processing_queue.join()
        else:
            for i, f in enumerate(files_to_process):
                logging.info(f"Processing {i+1}/{len(files_to_process)}: {os.path.basename(f)}")
                # Initialize tracker state for the file
                enqueued = tracker.enqueue_if_needed(f, source="batch-mode", valid_extensions=VALID_EXTENSIONS)
                # Claim it immediately
                if enqueued or tracker.file_states.get(os.path.abspath(f), {}).get("status") == tracker.STATUS_QUEUED:
                    attempt_id = tracker.claim_for_processing(f, worker_name="main")
                    if attempt_id:
                        success = smoothing_service.process_file(f, attempt_id, tracker, worker_name="main")
                        tracker.finish_processing(f, attempt_id, success, worker_name="main")

    logging.info("=== RINGKASAN ===")
    logging.info(f"Selesai")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logging.error("Fatal error. Detail:")
        logging.error(traceback.format_exc())
        raise
