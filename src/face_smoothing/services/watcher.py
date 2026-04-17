import os
import time
import logging
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class WatcherHandler(FileSystemEventHandler):
    """Event handler for directory watching."""

    def __init__(self, tracker, valid_extensions):
        self.tracker = tracker
        self.valid_extensions = valid_extensions

    def _wait_for_file_stabilization(self, path, timeout=10, interval=0.5):
        """Wait until file size stops changing and file is readable."""
        last_size = -1
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if not os.path.exists(path):
                    return False
                current_size = os.path.getsize(path)
                if current_size == last_size and current_size > 0:
                    # Size stabilized, try to open it
                    with open(path, 'rb'):
                        return True
                last_size = current_size
            except (IOError, OSError):
                # File locked or not ready
                pass
            time.sleep(interval)
        return False

    def _maybe_enqueue(self, src_path, source):
        img_path = os.path.abspath(src_path)
        # For watcher events, wait for file to be ready
        if source.startswith("watcher-"):
            if not self._wait_for_file_stabilization(img_path):
                logging.warning(f"[watcher] File tidak stabil/siap: {os.path.basename(img_path)}")
                return
        
        self.tracker.enqueue_if_needed(img_path, source=source, valid_extensions=self.valid_extensions)

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

    def on_modified(self, event):
        if event.is_directory:
            return
        self._maybe_enqueue(event.src_path, source="watcher-modified")

class HotFolderWatcher(Observer):
    """Watcher that monitors a folder for new face images."""

    def __init__(self, tracker, valid_extensions):
        super().__init__()
        self.tracker = tracker
        self.valid_extensions = valid_extensions
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
                logging.warning(
                    f"Error di watchdog thread (restart {self.restart_count}/{self.max_restarts}): {e}"
                )
                time.sleep(2)
                if self.should_keep_running():
                    self.run()
            else:
                logging.error(f"Maksimum restart tercapai. Menghentikan observer: {e}")
                raise

def periodic_stuck_checker(
    tracker, check_interval, processing_timeout_seconds, processed_ttl_seconds
):
    """Background loop to requeue stuck files and prune processed entries."""
    while True:
        time.sleep(check_interval)
        stuck_files = tracker.get_stuck_files(processing_timeout_seconds)
        if stuck_files:
            logging.warning(
                f"[tracker] Ditemukan {len(stuck_files)} file stuck, requeue..."
            )
            for filepath in stuck_files:
                tracker.reset_stuck_file(filepath, source="stuck-checker")
        pruned = tracker.prune_processed(processed_ttl_seconds)
        if pruned:
            logging.info(
                f"[tracker] Pruned {pruned} processed entries (ttl={processed_ttl_seconds}s)"
            )

def scan_existing_files(input_folder, valid_extensions):
    """Initial scan of input folder to enqueue existing images."""
    existing_files = []
    if os.path.exists(input_folder):
        for file in os.listdir(input_folder):
            filepath = os.path.join(input_folder, file)
            if os.path.isfile(filepath):
                _, ext = os.path.splitext(file)
                if ext.lower() in valid_extensions:
                    existing_files.append(os.path.abspath(filepath))
    return sorted(existing_files)
