import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

_PARSING_SESSION = None
_PARSING_INPUT_NAME = None
_PARSING_INPUT_SIZE = (512, 512)


def _resolve_parsing_model_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runtime_base = getattr(sys, "_MEIPASS", base_dir)
    candidate_paths = [
        os.path.join(runtime_base, "models", "faceparsing_resnet18.onnx"),
        os.path.join(base_dir, "models", "faceparsing_resnet18.onnx"),
    ]
    for model_path in candidate_paths:
        if os.path.isfile(model_path):
            return model_path
    raise RuntimeError(
        "BiSeNet parsing model not found: models/faceparsing_resnet18.onnx"
    )


def _get_parsing_session():
    global _PARSING_SESSION, _PARSING_INPUT_NAME, _PARSING_INPUT_SIZE
    if _PARSING_SESSION is not None:
        return _PARSING_SESSION, _PARSING_INPUT_NAME, _PARSING_INPUT_SIZE

    providers = ort.get_available_providers()
    session_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in providers
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(
        _resolve_parsing_model_path(), providers=session_providers
    )
    input_meta = session.get_inputs()[0]
    input_shape = input_meta.shape
    # Handle symbolic or missing dimensions gracefully
    input_h = 512
    input_w = 512
    if len(input_shape) >= 4:
        input_h = int(input_shape[2]) if isinstance(input_shape[2], int) else 512
        input_w = int(input_shape[3]) if isinstance(input_shape[3], int) else 512
    elif len(input_shape) >= 2:
        # Some models might have [batch, size, size, channels] or similar
        input_h = int(input_shape[1]) if isinstance(input_shape[1], int) else 512
        input_w = int(input_shape[2]) if len(input_shape) > 2 and isinstance(input_shape[2], int) else input_h

    _PARSING_SESSION = session
    _PARSING_INPUT_NAME = input_meta.name
    _PARSING_INPUT_SIZE = (input_w, input_h)
    return _PARSING_SESSION, _PARSING_INPUT_NAME, _PARSING_INPUT_SIZE


def _bisenet_parsing_outputs(aligned_face):
    session, input_name, (in_w, in_h) = _get_parsing_session()
    resized = cv2.resize(aligned_face, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Standard BiSeNet face parsing normalization.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb - mean) / std
    blob = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)

    outputs = session.run(None, {input_name: blob})
    logits = outputs[0]
    if logits.ndim == 4:
        parsing_map = np.argmax(logits[0], axis=0).astype(np.uint8)
    elif logits.ndim == 3:
        parsing_map = np.argmax(logits, axis=0).astype(np.uint8)
    else:
        raise RuntimeError(f"Unexpected BiSeNet output shape: {logits.shape}")

    # Keep only skin-ish classes for smoothing area.
    # 1=skin, 10=nose (exclude full nose later via exclusion boundary logic)
    skin_mask = np.isin(parsing_map, [1, 10]).astype(np.uint8) * 255
    parsing_map = cv2.resize(
        parsing_map,
        (aligned_face.shape[1], aligned_face.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    skin_mask = cv2.resize(
        skin_mask,
        (aligned_face.shape[1], aligned_face.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return parsing_map, skin_mask


def _opencv_cuda_available():
    try:
        if not hasattr(cv2, "cuda_GpuMat") or not hasattr(cv2, "cuda"):
            return False
        if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            return False
        if not hasattr(cv2.cuda, "Stream"):
            return False
        return True
    except Exception:
        return False


def _align_face(detected_img, face, target_size=256):
    bbox = face["bbox"]
    x1, y1, x2, y2 = bbox
    roi = detected_img[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None, None

    landmarks = face.get("landmarks")
    if landmarks is None or len(landmarks) != 5:
        return (
            cv2.resize(
                roi,
                (target_size, target_size),
                interpolation=cv2.INTER_LINEAR,
            ),
            None,
            None,
        )

    dst = np.array(
        [[96, 110], [160, 110], [128, 150], [102, 182], [154, 182]],
        dtype=np.float32,
    ) * (target_size / 256.0)

    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32),
        dst,
        method=cv2.LMEDS,
    )

    if matrix is None:
        return (
            cv2.resize(
                roi,
                (target_size, target_size),
                interpolation=cv2.INTER_LINEAR,
            ),
            None,
            landmarks.astype(np.float32),
        )

    aligned = cv2.warpAffine(
        detected_img,
        matrix,
        (target_size, target_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    aligned_landmarks = cv2.transform(
        landmarks.reshape(1, -1, 2).astype(np.float32), matrix
    ).reshape(-1, 2)

    return aligned, matrix, aligned_landmarks


def _nose_wing_boundary_mask(parsing_map, landmarks=None):
    h, w = parsing_map.shape[:2]
    nose_mask = (parsing_map == 10).astype(np.uint8) * 255
    if cv2.countNonZero(nose_mask) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(nose_mask, cv2.MORPH_GRADIENT, kernel)

    ys, xs = np.where(nose_mask > 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    nose_w = max(1, x_max - x_min + 1)
    nose_h = max(1, y_max - y_min + 1)
    cx = int((x_min + x_max) * 0.5)

    side_region = np.zeros_like(boundary)
    for y in range(y_min, y_max + 1):
        rel_y = (y - y_min) / max(1, nose_h - 1)
        if rel_y < 0.32 or rel_y > 0.96:
            continue
        for x in range(x_min, x_max + 1):
            if abs(x - cx) >= int(nose_w * 0.18):
                side_region[y, x] = 255

    alar_boundary = cv2.bitwise_and(boundary, side_region)

    # Tambahan tipis dari landmark hidung ke arah mulut untuk menutup garis sayap hidung.
    if landmarks is not None and len(landmarks) >= 5:
        right_eye, left_eye, nose, right_mouth, left_mouth = landmarks[:5]
        eye_dist = float(np.linalg.norm(np.array(right_eye) - np.array(left_eye)))
        line_thickness = int(max(2, eye_dist * 0.04))
        for mouth_pt in (right_mouth, left_mouth):
            start_pt = (
                int(nose[0] + (mouth_pt[0] - nose[0]) * 0.22),
                int(nose[1] + (mouth_pt[1] - nose[1]) * 0.10),
            )
            end_pt = (
                int(nose[0] + (mouth_pt[0] - nose[0]) * 0.55),
                int(nose[1] + (mouth_pt[1] - nose[1]) * 0.62),
            )
            cv2.line(alar_boundary, start_pt, end_pt, 255, thickness=line_thickness)

    alar_boundary = cv2.dilate(alar_boundary, kernel, iterations=1)
    return alar_boundary


def _build_exclusion_mask(mask_shape, parsing_map, landmarks=None, aligned_face=None):
    if parsing_map is None:
        empty = np.zeros(mask_shape, dtype=np.uint8)
        return empty, empty

    exclusion_classes = [2, 3, 4, 5, 6, 11, 12, 13]
    hard_exclusion = np.isin(parsing_map, exclusion_classes).astype(np.uint8) * 255

    nose_wing_exclusion = _nose_wing_boundary_mask(parsing_map, landmarks=landmarks)
    hard_exclusion = cv2.bitwise_or(hard_exclusion, nose_wing_exclusion)

    soft_exclusion = cv2.GaussianBlur(hard_exclusion, (9, 9), 0)
    return hard_exclusion.astype(np.uint8), soft_exclusion.astype(np.uint8)


def _build_hair_candidate_masks(mask_shape, landmarks):
    """
    Return:
        mustache_candidate: float mask 0..1
        beard_candidate: float mask 0..1
    """
    h, w = mask_shape[:2]
    mustache = np.zeros((h, w), dtype=np.uint8)
    beard = np.zeros((h, w), dtype=np.uint8)

    if landmarks is None or len(landmarks) < 5:
        return mustache.astype(np.float32), beard.astype(np.float32)

    right_eye, left_eye, nose, right_mouth, left_mouth = landmarks[:5]

    mouth_center = ((right_mouth + left_mouth) / 2.0).astype(np.float32)
    mouth_dist = float(np.linalg.norm(np.array(right_mouth) - np.array(left_mouth)))
    eye_dist = float(np.linalg.norm(np.array(right_eye) - np.array(left_eye)))

    # mustache: thin strip above upper lip
    mustache_cx = int(np.round((nose[0] * 0.35) + (mouth_center[0] * 0.65)))
    mustache_cy = int(np.round((nose[1] * 0.40) + (mouth_center[1] * 0.60)))

    mustache_rx = int(max(8, min(w * 0.16, mouth_dist * 0.34)))
    mustache_ry = int(max(3, min(h * 0.025, mustache_rx * 0.30)))

    cv2.ellipse(
        mustache,
        (mustache_cx, mustache_cy),
        (mustache_rx, mustache_ry),
        0,
        0,
        360,
        255,
        -1,
    )

    # jangan turun ke bibir
    lip_cut_y = int(np.round(mouth_center[1] - max(1, mustache_ry * 0.15)))
    mustache[max(lip_cut_y, 0) :, :] = 0

    # beard: lower jaw only
    beard_cx = int(np.round(mouth_center[0]))
    beard_cy = int(np.round(mouth_center[1] + h * 0.18))

    beard_rx = int(max(w * 0.20, mouth_dist * 0.52, eye_dist * 0.22))
    beard_ry = int(max(h * 0.10, beard_rx * 0.52))

    cv2.ellipse(beard, (beard_cx, beard_cy), (beard_rx, beard_ry), 0, 0, 360, 255, -1)

    beard_top_cut = int(np.round(mouth_center[1] + h * 0.03))
    beard[: max(beard_top_cut, 0), :] = 0

    mustache = cv2.GaussianBlur(mustache, (9, 9), 0).astype(np.float32) / 255.0
    beard = cv2.GaussianBlur(beard, (13, 13), 0).astype(np.float32) / 255.0

    return mustache, beard


def _structure_aware_map(aligned_face):
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)
    theta = np.arctan2(gy, gx)

    cos2 = np.cos(2.0 * theta)
    sin2 = np.sin(2.0 * theta)

    mean_cos = cv2.GaussianBlur(cos2, (0, 0), 2.2)
    mean_sin = cv2.GaussianBlur(sin2, (0, 0), 2.2)
    consistency = np.sqrt(np.maximum(0.0, mean_cos * mean_cos + mean_sin * mean_sin))

    mag_ref = float(np.percentile(magnitude, 95))
    mag_norm = magnitude / max(mag_ref, 1e-6)
    mag_norm = np.clip(mag_norm, 0.0, 1.0)

    structure = np.clip(mag_norm * consistency, 0.0, 1.0)
    structure = cv2.GaussianBlur(structure, (0, 0), 1.5)
    return structure.astype(np.float32)


def _darkness_map(aligned_face):
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    darkness = np.clip((blur.mean() - gray) * 2.2 + 0.5, 0.0, 1.0)
    darkness = cv2.GaussianBlur(darkness, (0, 0), 1.2)
    return darkness.astype(np.float32)


def _parsing_mask(cfg, aligned_face, landmarks):
    parsing_map, base_mask = _bisenet_parsing_outputs(aligned_face)
    base_mask = cv2.GaussianBlur(base_mask, (7, 7), 0)

    h, w = base_mask.shape[:2]
    if landmarks is None or len(landmarks) < 5:
        raise RuntimeError("5-point landmarks are required for smoothing.")

    # Use landmarks to define a facial region of interest (oval)
    face_region_mask = np.zeros(base_mask.shape, dtype=np.uint8)
    right_eye, left_eye, nose, right_mouth, left_mouth = landmarks[:5]

    # Calculate face center and dimensions based on landmarks
    center_x = int(
        np.mean([right_eye[0], left_eye[0], right_mouth[0], left_mouth[0], nose[0]])
    )
    center_y = int(
        np.mean([right_eye[1], left_eye[1], nose[1], right_mouth[1], left_mouth[1]])
        - h * 0.02
    )

    eye_dist = float(np.linalg.norm(np.array(right_eye) - np.array(left_eye)))
    mouth_dist = float(np.linalg.norm(np.array(right_mouth) - np.array(left_mouth)))

    face_rx = int(max(w * 0.32, eye_dist * 1.1, mouth_dist * 1.05))
    face_ry = int(max(h * 0.42, face_rx * 1.25))

    cv2.ellipse(
        face_region_mask, (center_x, center_y), (face_rx, face_ry), 0, 0, 360, 255, -1
    )
    face_region_mask = cv2.GaussianBlur(face_region_mask, (21, 21), 0)

    # Intersect skin color with facial region
    skin_mask = cv2.bitwise_and(base_mask, face_region_mask)

    # Build exclusion masks (eyes, brows, mouth, etc.)
    hard_exclusion_mask, soft_exclusion_mask = _build_exclusion_mask(
        base_mask.shape, parsing_map, landmarks=landmarks, aligned_face=aligned_face
    )

    clean_skin = cv2.bitwise_and(skin_mask, cv2.bitwise_not(soft_exclusion_mask))

    mustache_candidate, beard_candidate = _build_hair_candidate_masks(
        base_mask.shape, landmarks
    )

    structure_map = _structure_aware_map(aligned_face)
    darkness = _darkness_map(aligned_face)

    mustache_conf = np.clip(mustache_candidate * structure_map * darkness, 0.0, 1.0)
    beard_conf = np.clip(beard_candidate * structure_map * darkness, 0.0, 1.0)

    # mustache lebih ringan, beard lebih kuat
    mustache_atten = np.clip(1.0 - (mustache_conf * 0.38), 0.0, 1.0)
    beard_atten = np.clip(1.0 - (beard_conf * 0.72), 0.0, 1.0)

    attenuation = np.clip(mustache_atten * beard_atten, 0.0, 1.0)

    base_alpha = clean_skin.astype(np.float32) / 255.0
    adaptive_alpha = np.clip(base_alpha * attenuation, 0.0, 1.0)
    adaptive_alpha = cv2.GaussianBlur(adaptive_alpha, (0, 0), 1.0)

    # hard exclusion absolut
    adaptive_alpha[hard_exclusion_mask > 0] = 0.0

    return (adaptive_alpha * 255.0).astype(np.uint8), hard_exclusion_mask


def _effective_alpha(gray_mask, hard_exclusion_mask=None):
    alpha = gray_mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)
    alpha = np.clip((alpha - 0.12) / 0.88, 0.0, 1.0)
    alpha = np.power(alpha, 1.15)

    if hard_exclusion_mask is not None:
        alpha[hard_exclusion_mask > 0] = 0.0

    return alpha


def _smooth_with_mask(cfg, face_img, gray_mask, hard_exclusion_mask=None):
    alpha_gray = _effective_alpha(gray_mask, hard_exclusion_mask=hard_exclusion_mask)
    cuda_ok = _opencv_cuda_available()
    has_cuda_bilateral = hasattr(cv2.cuda, "bilateralFilter") or hasattr(
        cv2.cuda, "createBilateralFilter"
    )
    has_cuda_blend = hasattr(cv2.cuda, "blendLinear") or (
        hasattr(cv2.cuda, "multiply") and hasattr(cv2.cuda, "add")
    )

    if cuda_ok and has_cuda_bilateral and has_cuda_blend:
        stream = cv2.cuda.Stream()
        gpu_face = cv2.cuda_GpuMat()
        gpu_face.upload(face_img, stream)

        if hasattr(cv2.cuda, "bilateralFilter"):
            gpu_blurred = cv2.cuda.bilateralFilter(
                gpu_face,
                cfg["filter"]["diameter"],
                cfg["filter"]["sigma_1"],
                cfg["filter"]["sigma_2"],
                stream=stream,
            )
        else:
            bilateral = cv2.cuda.createBilateralFilter(
                gpu_face.type(),
                -1,
                cfg["filter"]["diameter"],
                cfg["filter"]["sigma_1"],
                cfg["filter"]["sigma_2"],
            )
            gpu_blurred = bilateral.apply(gpu_face, stream=stream)

        weight_1 = (1.0 - alpha_gray).astype(np.float32)
        weight_2 = alpha_gray.astype(np.float32)
        gpu_w1 = cv2.cuda_GpuMat()
        gpu_w2 = cv2.cuda_GpuMat()
        gpu_w1.upload(weight_1, stream)
        gpu_w2.upload(weight_2, stream)

        if hasattr(cv2.cuda, "blendLinear"):
            gpu_out = cv2.cuda.blendLinear(
                gpu_face, gpu_blurred, gpu_w1, gpu_w2, stream=stream
            )
        else:
            gpu_face_f = cv2.cuda_GpuMat()
            gpu_blurred_f = cv2.cuda_GpuMat()
            gpu_face.convertTo(cv2.CV_32F, gpu_face_f, stream=stream)
            gpu_blurred.convertTo(cv2.CV_32F, gpu_blurred_f, stream=stream)

            gpu_w1_3 = cv2.cuda.merge([gpu_w1, gpu_w1, gpu_w1], stream=stream)
            gpu_w2_3 = cv2.cuda.merge([gpu_w2, gpu_w2, gpu_w2], stream=stream)
            gpu_mix_1 = cv2.cuda.multiply(gpu_face_f, gpu_w1_3, stream=stream)
            gpu_mix_2 = cv2.cuda.multiply(gpu_blurred_f, gpu_w2_3, stream=stream)
            gpu_sum = cv2.cuda.add(gpu_mix_1, gpu_mix_2, stream=stream)
            gpu_out = cv2.cuda_GpuMat()
            gpu_sum.convertTo(cv2.CV_8U, gpu_out, stream=stream)

        blended = gpu_out.download(stream)
        stream.waitForCompletion()
        return blended

    # CPU fallback when OpenCV CUDA is unavailable.
    blurred = cv2.bilateralFilter(
        face_img,
        cfg["filter"]["diameter"],
        cfg["filter"]["sigma_1"],
        cfg["filter"]["sigma_2"],
    )
    alpha = cv2.merge([alpha_gray, alpha_gray, alpha_gray]).astype(np.float32)
    blended = (
        face_img.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _blend_full_frame_gpu(base_img, overlay_img, alpha_gray):
    cuda_ok = _opencv_cuda_available()
    has_cuda_blend = hasattr(cv2.cuda, "blendLinear") or (
        hasattr(cv2.cuda, "multiply") and hasattr(cv2.cuda, "add")
    )
    if cuda_ok and has_cuda_blend:
        stream = cv2.cuda.Stream()
        gpu_base = cv2.cuda_GpuMat()
        gpu_overlay = cv2.cuda_GpuMat()
        gpu_w2 = cv2.cuda_GpuMat()
        gpu_w1 = cv2.cuda_GpuMat()
        gpu_base.upload(base_img, stream)
        gpu_overlay.upload(overlay_img, stream)
        gpu_w2.upload(alpha_gray.astype(np.float32), stream)
        gpu_w1.upload((1.0 - alpha_gray).astype(np.float32), stream)

        if hasattr(cv2.cuda, "blendLinear"):
            gpu_out = cv2.cuda.blendLinear(
                gpu_base, gpu_overlay, gpu_w1, gpu_w2, stream=stream
            )
        else:
            gpu_base_f = cv2.cuda_GpuMat()
            gpu_overlay_f = cv2.cuda_GpuMat()
            gpu_base.convertTo(cv2.CV_32F, gpu_base_f, stream=stream)
            gpu_overlay.convertTo(cv2.CV_32F, gpu_overlay_f, stream=stream)
            gpu_w1_3 = cv2.cuda.merge([gpu_w1, gpu_w1, gpu_w1], stream=stream)
            gpu_w2_3 = cv2.cuda.merge([gpu_w2, gpu_w2, gpu_w2], stream=stream)
            gpu_mix_1 = cv2.cuda.multiply(gpu_base_f, gpu_w1_3, stream=stream)
            gpu_mix_2 = cv2.cuda.multiply(gpu_overlay_f, gpu_w2_3, stream=stream)
            gpu_sum = cv2.cuda.add(gpu_mix_1, gpu_mix_2, stream=stream)
            gpu_out = cv2.cuda_GpuMat()
            gpu_sum.convertTo(cv2.CV_8U, gpu_out, stream=stream)

        output_img = gpu_out.download(stream)
        stream.waitForCompletion()
        return output_img

    alpha = cv2.merge([alpha_gray, alpha_gray, alpha_gray]).astype(np.float32)
    output = (
        base_img.astype(np.float32) * (1.0 - alpha)
        + overlay_img.astype(np.float32) * alpha
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def smooth_face(cfg, detected_img, faces):
    if detected_img is None:
        return None, None, None, None
    output_img = detected_img.copy()
    h, w = detected_img.shape[:2]
    full_mask_gray = np.zeros((h, w), dtype=np.uint8)
    roi_img = None
    smoothed_roi = None

    for face in faces:
        bbox = face["bbox"]
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue

        aligned_face, matrix, aligned_landmarks = _align_face(detected_img, face)
        if aligned_face is None:
            continue

        parsing_mask, hard_exclusion_mask = _parsing_mask(
            cfg,
            aligned_face,
            aligned_landmarks,
        )

        smoothed_aligned = _smooth_with_mask(
            cfg,
            aligned_face,
            parsing_mask,
            hard_exclusion_mask=hard_exclusion_mask,
        )

        if matrix is not None:
            inv_matrix = cv2.invertAffineTransform(matrix)
            h_orig, w_orig = output_img.shape[:2]

            warped_face = cv2.warpAffine(
                smoothed_aligned,
                inv_matrix,
                (w_orig, h_orig),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            warped_mask = cv2.warpAffine(
                parsing_mask,
                inv_matrix,
                (w_orig, h_orig),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            warped_hard_exclusion = cv2.warpAffine(
                hard_exclusion_mask,
                inv_matrix,
                (w_orig, h_orig),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )

            alpha_gray = _effective_alpha(
                warped_mask,
                hard_exclusion_mask=warped_hard_exclusion,
            )
            output_img = _blend_full_frame_gpu(output_img, warped_face, alpha_gray)

            full_mask_gray = cv2.max(
                full_mask_gray,
                (alpha_gray * 255.0).astype(np.uint8),
            )

            roi_img = aligned_face
            smoothed_roi = smoothed_aligned

        else:
            roi_img = output_img[y1:y2, x1:x2]
            if roi_img.size == 0:
                continue

            resized_mask = cv2.resize(
                parsing_mask,
                (roi_img.shape[1], roi_img.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            resized_hard_exclusion = cv2.resize(
                hard_exclusion_mask,
                (roi_img.shape[1], roi_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            smoothed_roi = _smooth_with_mask(
                cfg,
                roi_img,
                resized_mask,
                hard_exclusion_mask=resized_hard_exclusion,
            )

            output_img[y1:y2, x1:x2] = smoothed_roi

            resized_alpha = _effective_alpha(
                resized_mask,
                hard_exclusion_mask=resized_hard_exclusion,
            )

            full_mask_gray[y1:y2, x1:x2] = cv2.max(
                full_mask_gray[y1:y2, x1:x2],
                (resized_alpha * 255.0).astype(np.uint8),
            )

    if roi_img is None:
        roi_img = detected_img.copy()
    if smoothed_roi is None:
        smoothed_roi = detected_img.copy()

    full_mask = cv2.merge([full_mask_gray, full_mask_gray, full_mask_gray])
    return output_img, roi_img, full_mask, smoothed_roi
