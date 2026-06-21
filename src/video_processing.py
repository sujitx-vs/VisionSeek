import os
import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel


VALID_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def load_models(
    yolo_model_path="../models/yolov8x.pt",
    siglip_model_name="google/siglip-base-patch16-224"
):
    """
    Load YOLO and SigLIP models once and return them.
    """
    siglip_model = AutoModel.from_pretrained(siglip_model_name)
    siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)

    yolo_model = YOLO(yolo_model_path)

    return yolo_model, siglip_model, siglip_processor


def embed_crop(crop, model, processor):
    """
    Generate normalized embedding for a cropped image.
    """
    inputs = processor(images=crop, return_tensors="pt")

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.squeeze(0).cpu().numpy()


def process_video(
    video_path,
    yolo_model,
    siglip_model,
    siglip_processor,
    frame_sample_rate=1,
    padding=20,
    save_embeddings_path=None,
    save_metadata_path=None
):
    """
    Process a single video:
    - sample frames
    - detect objects using YOLO
    - crop detected objects
    - embed crops using SigLIP
    - build metadata table

    Returns:
        metadata_df (pd.DataFrame)
        total_embd (np.ndarray)
    """

    metadata_rows = []
    total_embd = []
    processed_frame = 0
    object_id = 0
    cap = None

    if not video_path.lower().endswith(VALID_VIDEO_EXTS):
        raise ValueError(f"Unsupported video format: {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration_seconds = total_frames / fps if fps > 0 else 0
        fps_int = int(fps) if fps > 0 else 30

        print("--- VIDEO METADATA ---")
        print(f"File Path: {video_path}")
        print(f"Resolution: {width}x{height} pixels")
        print(f"Frame Rate: {fps:.2f} FPS")
        print(f"Total Frames: {total_frames} frames")
        print(f"Duration: {duration_seconds:.2f} seconds")

        # process 1 frame per second by default
        step = fps_int * frame_sample_rate

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp = frame_no / fps if fps > 0 else 0

            results = yolo_model(frame)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    frame_h, frame_w = frame.shape[:2]

                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame_w, x2 + padding)
                    y2 = min(frame_h, y2 + padding)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    vec = embed_crop(crop_rgb, siglip_model, siglip_processor)

                    total_embd.append(vec)

                    metadata_rows.append({
                        "id": object_id,
                        "frame_no": frame_no,
                        "timestamp": round(timestamp, 2),
                        "yolo_class": result.names[class_id],
                        "confidence": round(confidence, 4)
                    })

                    object_id += 1

            processed_frame += 1

            # skip remaining frames to sample at the required rate
            for _ in range(step - 1):
                success, _ = cap.read()
                if not success:
                    break

        total_embd = np.array(total_embd)
        metadata_df = pd.DataFrame(metadata_rows)

        print(f"Processed Frames: {processed_frame}")
        print(f"Embedding Shape: {total_embd.shape}")

        if save_embeddings_path:
            os.makedirs(os.path.dirname(save_embeddings_path), exist_ok=True)
            np.save(save_embeddings_path, total_embd)

        if save_metadata_path:
            os.makedirs(os.path.dirname(save_metadata_path), exist_ok=True)
            metadata_df.to_csv(save_metadata_path, index=False)

        return metadata_df, total_embd

    finally:
        if cap is not None:
            cap.release()