import cv2
import numpy as np
import pandas as pd
import torch

VALID_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

def embed_image_rgb(img_rgb, siglip_model, siglip_processor,device):
    """
    Generate normalized SigLIP embedding for an RGB image
    (can be full frame or crop).
    """
    inputs = siglip_processor(images=img_rgb, return_tensors="pt")
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

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

def vid_embd(tracked_objects,cap,siglip_model, siglip_processor,device):
    frame_embeddings = []
    crop_embeddings = []
    metadata =[]
    for track_id,track_info in tracked_objects.items():
        class_name = track_info["class_name"]

        for frame_info in track_info["best_frames"]:
            frame_no = frame_info["frame_no"]
            timestamp = frame_info["timestamp"]
            bbox = list(map(int, frame_info["bbox"]))
            confidence = frame_info["confidence"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                continue
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_emb = embed_image_rgb(frame_rgb, siglip_model, siglip_processor,device)
            frame_embeddings.append(frame_emb)


            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_emb = embed_image_rgb(crop_rgb,siglip_model,siglip_processor,device)
            crop_embeddings.append(crop_emb)


            metadata.append({
                "track_id": track_id,
                "class_name": class_name,
                "frame_no": frame_no,
                "timestamp": timestamp,
                "bbox": bbox,
                "confidence": confidence
            })  

    frame_embeddings = np.array(frame_embeddings,dtype=np.float32)
    crop_embeddings = np.array(crop_embeddings,dtype=np.float32)
    metadata_df = pd.DataFrame(metadata)

    return frame_embeddings,crop_embeddings,metadata_df