import os
import logging
from datetime import datetime, timedelta
from threading import Lock

class ProcessingTracker:
    """Intelligently tracks file processing states to avoid duplicates and handle retries."""
    
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

    def _get_mtime(self, filepath):
        try:
            return os.path.getmtime(filepath)
        except OSError:
            return None

    def enqueue_if_needed(self, filepath, source, valid_extensions):
        filepath = self._normalize_path(filepath)
        img_name = os.path.basename(filepath)
        _, ext = os.path.splitext(filepath)
        current_mtime = self._get_mtime(filepath)
        
        if ext.lower() not in valid_extensions:
            return False

        with self.lock:
            entry = self.file_states.get(filepath)
            status = entry["status"] if entry else None
            last_mtime = entry.get("last_mtime") if entry else None

            # If file was already processed but changed on disk, allow re-queue.
            if (
                status == self.STATUS_PROCESSED
                and current_mtime is not None
                and last_mtime is not None
                and current_mtime > last_mtime
            ):
                attempt_id = self._next_attempt_id(filepath)
                self.file_states[filepath] = {
                    "status": self.STATUS_QUEUED,
                    "updated_at": datetime.now(),
                    "attempt_id": attempt_id,
                    "source": source,
                    "last_mtime": current_mtime,
                }
                self.processing_queue.put(filepath)
                logging.info(
                    f"[tracker] Re-queued modified file ({source}, attempt={attempt_id}): {img_name}"
                )
                return True

            # If file changes while currently processing, mark to re-queue after finish.
            if (
                status == self.STATUS_PROCESSING
                and current_mtime is not None
                and last_mtime is not None
                and current_mtime > last_mtime
            ):
                entry["requeue_after_finish"] = True
                entry["pending_mtime"] = current_mtime
                logging.info(
                    f"[tracker] Marked for re-queue after finish ({source}): {img_name}"
                )
                return False

            if status in {
                self.STATUS_QUEUED,
                self.STATUS_PROCESSING,
                self.STATUS_PROCESSED,
            }:
                logging.info(
                    f"[tracker] Skip enqueue ({source}, status={status}): {img_name}"
                )
                return False

            attempt_id = self._next_attempt_id(filepath)
            self.file_states[filepath] = {
                "status": self.STATUS_QUEUED,
                "updated_at": datetime.now(),
                "attempt_id": attempt_id,
                "source": source,
                "last_mtime": current_mtime,
            }
            self.processing_queue.put(filepath)
            logging.info(
                f"[tracker] Queued ({source}, attempt={attempt_id}): {img_name}"
            )
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
                    "last_mtime": self._get_mtime(filepath),
                }
                logging.warning(
                    f"[{worker_name}] Claimed missing tracker entry: {img_name} (attempt={attempt_id})"
                )
                return attempt_id

            status = entry["status"]
            if status == self.STATUS_PROCESSED:
                logging.info(
                    f"[{worker_name}] Skip claim (status=processed): {img_name}"
                )
                return None
            if status != self.STATUS_QUEUED:
                logging.info(
                    f"[{worker_name}] Skip claim (status={status}): {img_name}"
                )
                return None

            entry["status"] = self.STATUS_PROCESSING
            entry["updated_at"] = datetime.now()
            entry["worker_name"] = worker_name
            latest_mtime = self._get_mtime(filepath)
            if latest_mtime is not None:
                entry["last_mtime"] = latest_mtime
            logging.info(
                f"[{worker_name}] Claimed: {img_name} (attempt={entry['attempt_id']})"
            )
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
                logging.info(
                    f"[{worker_name}] Finish ignored; missing entry: {img_name}"
                )
                return False
            if entry.get("attempt_id") != attempt_id:
                logging.info(
                    f"[{worker_name}] Finish ignored; stale attempt {attempt_id}: {img_name}"
                )
                return False

            if success:
                entry["status"] = self.STATUS_PROCESSED
                entry["updated_at"] = datetime.now()
                entry["worker_name"] = worker_name
                latest_mtime = self._get_mtime(filepath)
                if latest_mtime is not None:
                    entry["last_mtime"] = latest_mtime
                logging.info(f"[{worker_name}] Marked processed: {img_name}")
                if entry.get("requeue_after_finish"):
                    attempt_id = self._next_attempt_id(filepath)
                    entry["status"] = self.STATUS_QUEUED
                    entry["updated_at"] = datetime.now()
                    entry["attempt_id"] = attempt_id
                    entry["source"] = "requeue-after-finish"
                    pending_mtime = entry.pop("pending_mtime", None)
                    if pending_mtime is not None:
                        entry["last_mtime"] = pending_mtime
                    entry.pop("requeue_after_finish", None)
                    self.processing_queue.put(filepath)
                    logging.info(
                        f"[tracker] Re-queued modified file after finish (attempt={attempt_id}): {img_name}"
                    )
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
