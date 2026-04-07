import os
import sys
import logging
import traceback
LOG_PATH = os.path.join(os.path.expanduser("~"), "face-smoothing.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, encoding="utf-8")],
)

runtime_paths = [
    os.path.dirname(os.path.abspath(__file__)),
    getattr(sys, "_MEIPASS", ""),
    os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "_internal"),
]
for runtime_path in runtime_paths:
    if runtime_path and os.path.isdir(runtime_path) and runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)

try:
    import argparse
    import yaml
    import time
    import glob
    import cv2
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock, Thread, current_thread
    from queue import Queue, Empty
    from datetime import datetime, timedelta
    import gc
    from dotenv import load_dotenv
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    from detector.detect import detect_face
    from detector.smooth import smooth_face
    from utils.image import (load_image,
                             save_image,
                             save_steps,
                             check_img_size,
                             process_image,
                             check_if_adding_bboxes)
    from utils.video import (split_video,
                             process_video)
    from utils.types import (is_image,
                             is_video,
                             is_directory)

    load_dotenv()
except Exception:
    logging.error("Import error. Detail:")
    logging.error(traceback.format_exc())
    raise

# --- CONSTANTS AND HELPERS ---
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
CLEANUP_INTERVAL = 5
DEFAULT_PARALLEL_WORKERS = 3
DEFAULT_STUCK_CHECK_INTERVAL = 1800
DEFAULT_PROCESSING_TIMEOUT = 1800
DEFAULT_PROCESSED_TTL_SECONDS = 86400
FILE_READY_POLL_INTERVAL = 0.5
FILE_READY_MAX_POLLS = 20

def _resource_path(relative_path):
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, relative_path)

def _positive_int(value):
    try:
        int_value = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if int_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return int_value

def _get_env_int(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return _positive_int(raw_value)
    except argparse.ArgumentTypeError:
        logging.warning(f"Invalid {name}={raw_value!r}. Using default {default}.")
        return default

def _get_env_str(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    raw_value = raw_value.strip()
    return raw_value if raw_value else default

# --- PIPELINE TRACKER & WATCHER ---

class CustomObserver(Observer):
    def __init__(self):
        super().__init__()
        self.restart_count = 0
        self.max_restarts = 5

    def run(self):
        try:
            super().run()
        except KeyboardInterrupt:
            logging.info("File watcher dihentikan oleh user")
            raise
        except Exception as e:
            self.restart_count += 1
            if self.restart_count <= self.max_restarts:
                logging.warning(f"Error di watchdog thread (restart {self.restart_count}/{self.max_restarts}): {e}")
                time.sleep(2)
                if self.should_keep_running():
                    self.run()
            else:
                logging.error(f"Maksimum restart tercapai. Menghentikan observer: {e}")
                raise

class ProcessingTracker:
    STATUS_QUEUED = "queued"
    STATUS_PROCESSING = "processing"
    STATUS_PROCESSED = "processed"

    def __init__(self, processing_queue):
        self.lock = Lock()
        self.processing_queue = processing_queue
        self.file_states = {}
        self.attempt_counters = {}

    def _normalize_path(self, filepath):
        return os.path.abspath(filepath)

    def _next_attempt_id(self, filepath):
        current_attempt = self.attempt_counters.get(filepath, 0) + 1
        self.attempt_counters[filepath] = current_attempt
        return current_attempt

    def enqueue_if_needed(self, filepath, source):
        filepath = self._normalize_path(filepath)
        img_name = os.path.basename(filepath)
        with self.lock:
            entry = self.file_states.get(filepath)
            status = entry["status"] if entry else None
            if status in {self.STATUS_QUEUED, self.STATUS_PROCESSING, self.STATUS_PROCESSED}:
                logging.info(f"[tracker] Skip enqueue ({source}, status={status}): {img_name}")
                return False

            attempt_id = self._next_attempt_id(filepath)
            self.file_states[filepath] = {
                "status": self.STATUS_QUEUED,
                "updated_at": datetime.now(),
                "attempt_id": attempt_id,
                "source": source,
            }
            self.processing_queue.put(filepath)
            logging.info(f"[tracker] Queued ({source}, attempt={attempt_id}): {img_name}")
            return True

    def claim_for_processing(self, filepath, worker_name):
        filepath = self._normalize_path(filepath)
        img_name = os.path.basename(filepath)
        with self.lock:
            entry = self.file_states.get(filepath)
            if entry is None:
                attempt_id = self._next_attempt_id(filepath)
                self.file_states[filepath] = {
                    "status": self.STATUS_PROCESSING,
                    "updated_at": datetime.now(),
                    "attempt_id": attempt_id,
                    "worker_name": worker_name,
                    "source": "queue-recovery",
                }
                logging.warning(f"[{worker_name}] Claimed missing tracker entry: {img_name} (attempt={attempt_id})")
                return attempt_id

            status = entry["status"]
            if status == self.STATUS_PROCESSED:
                logging.info(f"[{worker_name}] Skip claim (status=processed): {img_name}")
                return None
            if status != self.STATUS_QUEUED:
                logging.info(f"[{worker_name}] Skip claim (status={status}): {img_name}")
                return None

            entry["status"] = self.STATUS_PROCESSING
            entry["updated_at"] = datetime.now()
            entry["worker_name"] = worker_name
            logging.info(f"[{worker_name}] Claimed: {img_name} (attempt={entry['attempt_id']})")
            return entry["attempt_id"]

    def is_current_attempt(self, filepath, attempt_id):
        filepath = self._normalize_path(filepath)
        with self.lock:
            entry = self.file_states.get(filepath)
            if entry is None:
                return False
            return entry.get("attempt_id") == attempt_id

    def finish_processing(self, filepath, attempt_id, success, worker_name):
        filepath = self._normalize_path(filepath)
        img_name = os.path.basename(filepath)
        with self.lock:
            entry = self.file_states.get(filepath)
            if entry is None:
                logging.info(f"[{worker_name}] Finish ignored; missing entry: {img_name}")
                return False
            if entry.get("attempt_id") != attempt_id:
                logging.info(f"[{worker_name}] Finish ignored; stale attempt {attempt_id}: {img_name}")
                return False

            if success:
                entry["status"] = self.STATUS_PROCESSED
                entry["updated_at"] = datetime.now()
                entry["worker_name"] = worker_name
                logging.info(f"[{worker_name}] Marked processed: {img_name}")
            else:
                del self.file_states[filepath]
                logging.info(f"[{worker_name}] Released for retry: {img_name}")
            return True

    def get_stuck_files(self, timeout_seconds):
        with self.lock:
            now = datetime.now()
            stuck = []
            for filepath, entry in list(self.file_states.items()):
                if entry["status"] != self.STATUS_PROCESSING:
                    continue
                if (now - entry["updated_at"]).total_seconds() > timeout_seconds:
                    stuck.append(filepath)
            return stuck

    def prune_processed(self, ttl_seconds):
        if ttl_seconds is None:
            return 0
        cutoff = datetime.now() - timedelta(seconds=ttl_seconds)
        with self.lock:
            pruned = 0
            for filepath, entry in list(self.file_states.items()):
                if entry["status"] != self.STATUS_PROCESSED:
                    continue
                updated_at = entry.get("updated_at")
                if updated_at is None or updated_at >= cutoff:
                    continue
                del self.file_states[filepath]
                self.attempt_counters.pop(filepath, None)
                pruned += 1
            return pruned

    def reset_stuck_file(self, filepath, source):
        filepath = self._normalize_path(filepath)
        img_name = os.path.basename(filepath)
        with self.lock:
            entry = self.file_states.get(filepath)
            if entry is None or entry["status"] != self.STATUS_PROCESSING:
                return False

            attempt_id = self._next_attempt_id(filepath)
            self.file_states[filepath] = {
                "status": self.STATUS_QUEUED,
                "updated_at": datetime.now(),
                "attempt_id": attempt_id,
                "source": source,
            }
            self.processing_queue.put(filepath)
            logging.warning(f"[tracker] Re-queued stuck file ({source}): {img_name}")
            return True

def periodic_stuck_checker(tracker, check_interval, processing_timeout_seconds, processed_ttl_seconds):
    while True:
        time.sleep(check_interval)
        stuck_files = tracker.get_stuck_files(processing_timeout_seconds)
        if stuck_files:
            logging.warning(f"[tracker] Ditemukan {len(stuck_files)} file stuck, requeue...")
            for filepath in stuck_files:
                tracker.reset_stuck_file(filepath, source="stuck-checker")
        pruned = tracker.prune_processed(processed_ttl_seconds)
        if pruned:
            logging.info(f"[tracker] Pruned {pruned} processed entries (ttl={processed_ttl_seconds}s)")

def scan_existing_files(input_folder):
    existing_files = []
    if os.path.exists(input_folder):
        for file in os.listdir(input_folder):
            filepath = os.path.join(input_folder, file)
            if os.path.isfile(filepath):
                _, ext = os.path.splitext(file)
                if ext.lower() in VALID_EXTENSIONS:
                    existing_files.append(os.path.abspath(filepath))
    return sorted(existing_files)

class MyHandler(FileSystemEventHandler):
    def __init__(self, tracker):
        self.tracker = tracker

    def _maybe_enqueue(self, src_path, source):
        img_path = os.path.abspath(src_path)
        _, ext = os.path.splitext(img_path)
        if ext.lower() not in VALID_EXTENSIONS:
            return
        self.tracker.enqueue_if_needed(img_path, source=source)

    def on_created(self, event):
        if event.is_directory:
            return
        self._maybe_enqueue(event.src_path, source="watcher-created")

    def on_moved(self, event):
        if event.is_directory:
            return
        dest_path = getattr(event, "dest_path", None)
        if not dest_path:
            return
        self._maybe_enqueue(dest_path, source="watcher-moved")

# --- FACE SMOOTHING LOGIC ---

def load_configs():
    cfg_path = _resource_path(os.path.join("configs", "configs.yaml"))
    with open(cfg_path, "r", encoding="utf-8") as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    cfg["net"]["model_file"] = _resource_path(cfg["net"]["model_file"])
    cfg["net"]["cfg_file"] = _resource_path(cfg["net"]["cfg_file"])
    return cfg

def process_single_image(cfg, net, img_path, attempt_id, args, output_path, tracker, enhance_lock=None, worker_name="worker"):
    success = False
    img_path = os.path.abspath(img_path)
    img_name = os.path.basename(img_path)

    try:
        start_time = time.time()
        
        # Wait for file to be fully copied
        file_size = -1
        timeout_count = 0
        file_is_stable = False
        while timeout_count < FILE_READY_MAX_POLLS:
            current_size = os.path.getsize(img_path)
            if current_size > 0 and current_size == file_size:
                file_is_stable = True
                break
            file_size = current_size
            time.sleep(FILE_READY_POLL_INTERVAL)
            timeout_count += 1

        if not file_is_stable:
            logging.error(f"[{worker_name}] Timeout menunggu file stabil: {img_name}")
            return False

        input_img = load_image(img_path)
        if input_img is None:
            logging.error(f"[{worker_name}] Gagal membaca gambar: {img_path}")
            return False

        logging.info(f"[{worker_name}] Processing: {img_name} (attempt={attempt_id})")

        if enhance_lock is not None:
            with enhance_lock:
                if not tracker.is_current_attempt(img_path, attempt_id):
                    return False
                img_steps = process_image(input_img, cfg, net)
        else:
            img_steps = process_image(input_img, cfg, net)

        if not tracker.is_current_attempt(img_path, attempt_id):
            return False

        output_img = check_if_adding_bboxes(args, img_steps)
        
        basename, ext = os.path.splitext(img_name)
        save_path = os.path.join(output_path, f"{basename}{ext}")
        os.makedirs(output_path, exist_ok=True)
        
        save_image(save_path, output_img)
        
        if args.save_steps:
            output_height = cfg['image']['img_steps_height']
            steps_filename = os.path.join(output_path, f"combined_{basename}{ext}")
            save_steps(steps_filename, img_steps, output_height)

        processing_time = time.time() - start_time
        logging.info(f"[{worker_name}] Saved: {img_name} -> {os.path.basename(save_path)} ({processing_time:.2f}s)")
        success = True
        return True

    except FileNotFoundError:
        logging.error(f"[{worker_name}] File hilang saat diproses: {img_path}")
        return False
    except Exception as e:
        logging.error(f"[{worker_name}] Error processing {img_path}: {e}")
        return False
    finally:
        tracker.finish_processing(img_path, attempt_id, success=success, worker_name=worker_name)

# --- MAIN ---

def parse_args():
    default_parallel_workers = _get_env_int("HOTFOLDER_WORKERS", DEFAULT_PARALLEL_WORKERS)
    default_processing_timeout = _get_env_int("PROCESSING_TIMEOUT_SECONDS", DEFAULT_PROCESSING_TIMEOUT)
    default_stuck_check_interval = _get_env_int("STUCK_CHECK_INTERVAL_SECONDS", DEFAULT_STUCK_CHECK_INTERVAL)
    default_processed_ttl = _get_env_int("PROCESSED_TTL_SECONDS", DEFAULT_PROCESSED_TTL_SECONDS)
    default_watch_input = _get_env_str(
        "HOTFOLDER_INPUT_DIR",
        os.path.join(os.path.expanduser("~"), "selfy-time", "beauty_input"),
    )
    default_watch_output = _get_env_str(
        "HOTFOLDER_OUTPUT_DIR",
        os.path.join(os.path.expanduser("~"), "selfy-time", "beauty_output"),
    )

    parser = argparse.ArgumentParser(description='Facial detection and smoothing pipeline.')
    parser.add_argument('--input', type=str, default=None, help='Input file or folder (Batch mode). If empty, runs Watcher mode.')
    parser.add_argument('--output', type=str, default=None, help='Output folder')
    parser.add_argument('--show-detections', action='store_true', help='Displays bounding boxes during inference.')
    parser.add_argument('--save-steps', action='store_true', help='Saves each step of the image.')
    parser.add_argument('--parallel-workers', type=_positive_int, default=default_parallel_workers, help="Jumlah worker.")
    parser.add_argument('--processing-timeout-seconds', type=_positive_int, default=default_processing_timeout)
    parser.add_argument('--stuck-check-interval-seconds', type=_positive_int, default=default_stuck_check_interval)
    parser.add_argument('--processed-ttl-seconds', type=_positive_int, default=default_processed_ttl)
    parser.add_argument('--watch-input', type=str, default=default_watch_input)
    parser.add_argument('--watch-output', type=str, default=default_watch_output)
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.input and not args.output:
        args.output = "data/output"
        logging.info(f"Output directory tidak ditentukan, menggunakan default: {args.output}")
        
    cfg = load_configs()
    net = cv2.dnn.readNetFromTensorflow(cfg['net']['model_file'], cfg['net']['cfg_file'])

    if args.input:
        # BATCH MODE
        input_path = args.input
        if is_video(input_path):
            process_video(input_path, args, cfg, net)
            logging.info("Video processing selesai.")
            return

        if is_image(input_path):
            img_list = [input_path]
        elif is_directory(input_path):
            img_list = sorted(glob.glob(os.path.join(input_path, "*")))
        else:
            logging.error("Input must be a valid image, video, or directory.")
            return

        os.makedirs(args.output, exist_ok=True)
        logging.info(f"Mode batch processing: {len(img_list)} file(s) ditemukan")

        success_count = 0
        fail_count = 0
        
        # We can use ThreadPoolExecutor for batch mode just like GFPGAN script, 
        # but GFPGAN batch mode is sequential. We will keep it simple here.
        for i, img_path in enumerate(img_list):
            if not is_image(img_path):
                continue
            logging.info(f"Processing {i + 1}/{len(img_list)}: {os.path.basename(img_path)}")
            try:
                input_img = load_image(img_path)
                img_steps = process_image(input_img, cfg, net)
                output_img = check_if_adding_bboxes(args, img_steps)
                
                basename, ext = os.path.splitext(os.path.basename(img_path))
                save_path = os.path.join(args.output, f"{basename}{ext}")
                save_image(save_path, output_img)
                
                if args.save_steps:
                    steps_filename = os.path.join(args.output, f"combined_{basename}{ext}")
                    save_steps(steps_filename, img_steps, cfg['image']['img_steps_height'])
                
                success_count += 1
            except Exception as e:
                logging.error(f"Error: {e}")
                fail_count += 1

        logging.info(f"=== RINGKASAN ===\nBerhasil: {success_count}, Gagal: {fail_count}")

    else:
        # WATCHER MODE
        path = args.watch_input
        output_path = args.output if args.output else args.watch_output

        try:
            os.makedirs(path, exist_ok=True)
            os.makedirs(output_path, exist_ok=True)

            logging.info("=== FILE WATCHER MODE (PARALLEL) ===")
            logging.info(f"Monitoring: {path}")
            logging.info(f"Output: {output_path}")
            logging.info(f"Workers: {args.parallel_workers}")
            logging.info("=====================================")

            processing_queue = Queue()
            tracker = ProcessingTracker(processing_queue)
            enhance_lock = Lock()

            event_handler = MyHandler(tracker)
            observer = CustomObserver()

            stuck_checker_thread = Thread(
                target=periodic_stuck_checker,
                args=(tracker, args.stuck_check_interval_seconds, args.processing_timeout_seconds, args.processed_ttl_seconds),
                daemon=True,
                name="stuck-checker",
            )
            stuck_checker_thread.start()

            observer.schedule(event_handler, path, recursive=False)
            observer.start()

            existing_files = scan_existing_files(path)
            if existing_files:
                logging.info(f"[tracker] Ditemukan {len(existing_files)} file existing")
                for filepath in existing_files:
                    tracker.enqueue_if_needed(filepath, source="startup-scan")

            def worker_thread():
                worker_name = current_thread().name
                while True:
                    try:
                        img_path = processing_queue.get(timeout=1)
                        try:
                            if img_path is None:
                                return
                            attempt_id = tracker.claim_for_processing(img_path, worker_name)
                            if attempt_id is None:
                                continue

                            process_single_image(
                                cfg, net, img_path, attempt_id, args, output_path, tracker, enhance_lock, worker_name
                            )
                        finally:
                            processing_queue.task_done()
                    except Empty:
                        continue
                    except Exception as e:
                        logging.error(f"[{worker_name}] Worker error: {e}")

            workers = []
            for i in range(args.parallel_workers):
                worker = Thread(target=worker_thread, daemon=True, name=f"worker-{i + 1}")
                worker.start()
                workers.append(worker)

            logging.info("Tekan Ctrl+C untuk menghentikan")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("Menghentikan...")
                for _ in range(args.parallel_workers):
                    processing_queue.put(None)
                for worker in workers:
                    worker.join(timeout=5)
                observer.stop()
                observer.join(timeout=5)

        except Exception as e:
            logging.error(f"Error: {e}")

    logging.info("Selesai")

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logging.error("Fatal error. Detail:")
        logging.error(traceback.format_exc())
        raise
