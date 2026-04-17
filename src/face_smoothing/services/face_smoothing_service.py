import os
import time
import logging
import traceback
from threading import Lock
from face_smoothing.utils.image import load_image, process_image, save_image, save_steps, check_img_size, check_if_adding_bboxes
from face_smoothing.utils.video import process_video
from face_smoothing.utils.types import is_image, is_video

class FaceSmoothingService:
    """Service layer that orchestrates the face smoothing pipeline for individual files."""

    def __init__(self, cfg, detector, args):
        self.cfg = cfg
        self.detector = detector
        self.args = args
        self.lock = Lock()

    def _load_image_with_retry(self, img_path, worker_name, retries=12, delay_seconds=0.25):
        """Retry image loading to handle files that are still being copied into watched folder."""
        last_exists = False
        for attempt in range(1, retries + 1):
            last_exists = os.path.isfile(img_path)
            if last_exists:
                img = load_image(img_path)
                if img is not None:
                    return img
            if attempt < retries:
                time.sleep(delay_seconds)
        logging.error(
            f"[{worker_name}] Gagal memuat gambar setelah {retries} percobaan: {os.path.basename(img_path)} (exists={last_exists})"
        )
        return None

    def process_file(self, img_path, attempt_id, tracker, worker_name="worker"):
        """Process a single image or video file."""
        success = False
        img_path = os.path.abspath(img_path)
        img_name = os.path.basename(img_path)

        try:
            start_time = time.time()
            if is_video(img_path):
                logging.info(f"[{worker_name}] Memproses video: {img_name}")
                with self.lock:
                    process_video(img_path, self.args, self.cfg, self.detector)
                success = True
            elif is_image(img_path):
                input_img = self._load_image_with_retry(img_path, worker_name)
                if input_img is None:
                    logging.error(f"[{worker_name}] Gagal memuat gambar: {img_name}")
                    return False

                # Pipeline orchestration
                with self.lock:
                    img_steps = process_image(input_img, self.cfg, self.detector)
                (
                    _,
                    detected_img,
                    roi_img,
                    hsv_mask,
                    smoothed_roi,
                    output_w_bboxes,
                    output_img,
                ) = img_steps

                # Save results
                final_output = check_if_adding_bboxes(self.args, img_steps)
                
                # Check tracker before saving
                if not tracker.is_current_attempt(img_path, attempt_id):
                    logging.warning(f"[{worker_name}] Stale attempt for {img_name}, skipping save.")
                    return False

                out_name = self.cfg["image"]["output"] + img_name
                out_path = os.path.join(self.args.output, out_name)
                save_image(out_path, final_output)

                if self.args.save_detection:
                    det_path = os.path.join(self.args.output, "det_" + img_name)
                    save_image(det_path, detected_img)

                if self.args.save_parsing:
                    parsing_path = os.path.join(self.args.output, os.path.splitext(img_name)[0] + "_parsing_mask.jpg")
                    save_image(parsing_path, hsv_mask)
                    area_path = os.path.join(self.args.output, os.path.splitext(img_name)[0] + "_parsing_area.jpg")
                    save_image(area_path, roi_img)

                if self.args.save_steps:
                    steps_name = self.cfg["image"]["output_steps"] + img_name
                    steps_path = os.path.join(self.args.output, steps_name)
                    save_steps(steps_path, img_steps, self.cfg["image"]["img_steps_height"])

                success = True
                elapsed = time.time() - start_time
                logging.info(f"[{worker_name}] Berhasil: {img_name} ({elapsed:.2f}s)")

        except Exception as e:
            logging.error(f"[{worker_name}] Gagal memproses {img_name}: {e}")
            logging.error(traceback.format_exc())
            success = False

        return success
