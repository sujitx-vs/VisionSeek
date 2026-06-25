import os
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel


VALID_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def load_models(
    yolo_model_path="models/yolov8x.pt",
    siglip_model_name="google/siglip-base-patch16-224"
):
    """
    Load YOLO and SigLIP once.
    """
    yolo_model = YOLO(yolo_model_path)
    siglip_model = AutoModel.from_pretrained(siglip_model_name)
    siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)
    return yolo_model, siglip_model, siglip_processor


def embed_image_rgb(img_rgb, siglip_model, siglip_processor):
    """
    Generate normalized SigLIP embedding for an RGB image
    (can be full frame or crop).
    """
    inputs = siglip_processor(images=img_rgb, return_tensors="pt")

    with torch.no_grad():
        feats = siglip_model.get_image_features(**inputs)

        # extract actual tensor from model output
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif hasattr(feats, "image_embeds"):
            feats = feats.image_embeds

        # L2 normalize
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.squeeze(0).cpu().numpy()


def process_video_with_tracking(
    video_path,
    yolo_model,
    siglip_model,
    siglip_processor,
    sample_interval_sec=1,
    conf_threshold=0.5,
    padding=20,
    tracker_cfg="bytetrack.yaml",
    save_crop_emb_path=None,
    save_frame_emb_path=None,
    save_metadata_path=None
):
    """
    Process video using:
    - cv2 video loading
    - YOLO + ByteTrack on sampled frames
    - SigLIP embeddings for:
        1) full frame
        2) object crop

    STORE ONLY NEW TRACKS:
    - If a track_id appears for the first time -> save metadata + embeddings
    - If same track_id appears again in later frames -> skip

    Returns:
        metadata_df
        crop_embeddings   : np.ndarray [N, D]
        frame_embeddings  : np.ndarray [N, D]
    """

    if not video_path.lower().endswith(VALID_VIDEO_EXTS):
        raise ValueError(f"Unsupported video format: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    duration_seconds = total_frames / fps
    step = max(1, int(round(fps * sample_interval_sec)))
    # step = 2

    print("----- VIDEO INFO -----")
    print(f"Path: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration_seconds:.2f} sec")
    print(f"Sampling every {sample_interval_sec} second(s)")

    metadata_rows = []
    crop_embeddings = []
    frame_embeddings = []

    seen_track_ids = set()
    row_id = 0
    processed_frames = 0

    current_frame = 0

    try:
        while current_frame < total_frames:
            # jump directly to sampled frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            frame_no = current_frame
            timestamp_sec = frame_no / fps

            hrs = int(timestamp_sec // 3600)
            mins = int((timestamp_sec % 3600) // 60)
            secs = int(timestamp_sec % 60)
            timestamp_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

            # YOLO + ByteTrack on this sampled frame
            results = yolo_model.track(
                frame,
                persist=True,
                tracker=tracker_cfg,
                verbose=False
            )

            processed_frames += 1

            if not results:
                current_frame += step
                continue

            result = results[0]

            # if no boxes at all in this frame
            if result.boxes is None or len(result.boxes) == 0:
                current_frame += step
                continue

            # frame embedding will be created only if at least one NEW object is found
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_emb_cache = None

            for box in result.boxes:
                # track id may be missing in some cases
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                if confidence < conf_threshold:
                    continue

                # If already seen before, skip storing
                if track_id in seen_track_ids:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                # embed crop
                crop_vec = embed_image_rgb(crop_rgb, siglip_model, siglip_processor)
                crop_emb_index = len(crop_embeddings)
                crop_embeddings.append(crop_vec)

                # embed full frame ONLY once for this frame,
                # but reuse if multiple new objects appear in same frame
                if frame_emb_cache is None:
                    frame_emb_cache = embed_image_rgb(frame_rgb, siglip_model, siglip_processor)
                    frame_emb_index = len(frame_embeddings)
                    frame_embeddings.append(frame_emb_cache)
                else:
                    # same frame embedding reused
                    frame_emb_index = len(frame_embeddings) - 1

                metadata_rows.append({
                    "row_id": row_id,
                    "track_id": track_id,
                    "frame_no": frame_no,
                    "timestamp": timestamp_str,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "yolo_class": result.names[class_id],
                    "confidence": round(confidence, 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "crop_emb_index": crop_emb_index,
                    "frame_emb_index": frame_emb_index
                })

                seen_track_ids.add(track_id)
                row_id += 1

            current_frame += step

        crop_embeddings = np.array(crop_embeddings, dtype=np.float32)
        frame_embeddings = np.array(frame_embeddings, dtype=np.float32)
        metadata_df = pd.DataFrame(metadata_rows)

        print("\n----- PROCESSING SUMMARY -----")
        print(f"Processed sampled frames: {processed_frames}")
        print(f"Unique tracked objects stored: {len(metadata_df)}")
        print(f"Crop embeddings shape: {crop_embeddings.shape}")
        print(f"Frame embeddings shape: {frame_embeddings.shape}")

        # save outputs if paths provided
        if save_crop_emb_path:
            os.makedirs(os.path.dirname(save_crop_emb_path), exist_ok=True)
            np.save(save_crop_emb_path, crop_embeddings)

        if save_frame_emb_path:
            os.makedirs(os.path.dirname(save_frame_emb_path), exist_ok=True)
            np.save(save_frame_emb_path, frame_embeddings)

        if save_metadata_path:
            os.makedirs(os.path.dirname(save_metadata_path), exist_ok=True)
            metadata_df.to_csv(save_metadata_path, index=False)

        return metadata_df, crop_embeddings, frame_embeddings

    finally:
        cap.release()