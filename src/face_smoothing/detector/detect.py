import cv2
import numpy as np
import logging

from face_smoothing.utils import image


def detect_face(cfg, detector, input_img):
    if input_img is None:
        return None, []
    img_height, img_width = image.get_height_and_width(input_img)
    if img_height == 0 or img_width == 0:
        return input_img, []

    detected_img = input_img.copy()
    faces = []

    # Ensure image is C-contiguous and correct type for ONNX/GPU
    img_for_det = np.ascontiguousarray(input_img)

    max_retries = 2
    detected_faces = None
    last_error = None

    for attempt in range(max_retries):
        try:
            detected_faces = detector.get(img_for_det)
            break  # Success
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Slight wait and retry, sometimes helps with transient GPU state issues
                import time
                time.sleep(0.1)
                continue
    
    if detected_faces is None:
        if last_error:
            logging.warning(
                f"InsightFace detector.get failed after {max_retries} attempts; using detection-only fallback: {last_error}"
            )
        # Fallback: use low-level detector model only (without landmark/attribute models)
        # so one bad landmark inference does not drop the whole image.
        detected_faces = _fallback_detect_only(detector, img_for_det)
        if not detected_faces:
            logging.warning("InsightFace fallback detector found no faces.")
            return detected_img, faces

    if not detected_faces:
        return detected_img, faces

    conf_threshold = float(cfg["net"]["conf_threshold"])
    fallback_threshold = min(conf_threshold, 0.35)

    selected_faces = [
        face for face in detected_faces if float(face.det_score) >= conf_threshold
    ]
    if not selected_faces:
        selected_faces = [
            face
            for face in detected_faces
            if float(face.det_score) >= fallback_threshold
        ]

    for face in selected_faces:
        confidence = float(face.det_score)

        x1, y1, x2, y2 = face.bbox.astype(np.int32).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)

        raw_landmarks = np.array(face.kps, dtype=np.float32)
        if raw_landmarks.ndim != 2 or raw_landmarks.shape[0] < 5:
            continue
        raw_landmarks = raw_landmarks[:5]
        raw_landmarks[:, 0] = np.clip(raw_landmarks[:, 0], 0, img_width - 1)
        raw_landmarks[:, 1] = np.clip(raw_landmarks[:, 1], 0, img_height - 1)

        faces.append(
            {
                "bbox": [x1, y1, x2, y2],
                "landmarks": raw_landmarks,
                "score": confidence,
            }
        )

        cv2.rectangle(
            detected_img,
            (x1, y1),
            (x2, y2),
            cfg["image"]["bbox_color"],
            2,
        )

    return detected_img, faces


def _fallback_detect_only(detector, input_img):
    det_model = getattr(detector, "det_model", None)
    if det_model is None:
        return []

    try:
        # InsightFace detect returns (bboxes, kpss)
        bboxes, kpss = det_model.detect(input_img, max_num=0, metric="default")
    except Exception as error:
        logging.error(f"InsightFace fallback det_model.detect failed: {error}")
        return []

    if bboxes is None or len(bboxes) == 0:
        return []

    faces = []
    for i, bbox_row in enumerate(bboxes):
        bbox = np.array(bbox_row[:4], dtype=np.float32)
        score = float(bbox_row[4]) if len(bbox_row) > 4 else 0.0
        face_dict = {"bbox": bbox, "det_score": score}
        if kpss is not None and i < len(kpss) and kpss[i] is not None:
            face_dict["kps"] = np.array(kpss[i], dtype=np.float32)
        faces.append(type("FallbackFace", (), face_dict)())
    return faces
