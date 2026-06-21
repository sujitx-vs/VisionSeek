import cv2
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel


siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

os.makedirs("../models", exist_ok=True)

model_path = "../models/yolov8x.pt"
yolo_model = YOLO(model_path)

VID_PATH = "data/video_samples"

vid_folder = os.listdir(VID_PATH)

valid_video_exts = (".mp4",".avi",".mov",".mkv")

vid_file = os.path.join(VID_PATH,vid_folder[0])

metadata_rows = []
total_embd = []

def embed_crop(crop,model,processor):
    inputs = processor(
    images = crop,
    return_tensors = "pt"
    )
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.squeeze(0).cpu().numpy()
    
        



cap = None
processed_frame = 0

try:
    if vid_file.lower().endswith(valid_video_exts):
        cap = cv2.VideoCapture(vid_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if cap.isOpened():
            print("Valid video")    
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            duration_seconds = total_frames / fps if fps > 0 else 0

            print("--- VIDEO METADATA ---")
            print(f"File Path:  {vid_file}")
            print(f"Resolution: {width}x{height} pixels")
            print(f"Frame Rate: {fps:.2f} FPS")
            print(f"Total Frames: {total_frames} frames")
            print(f"Duration:   {duration_seconds:.2f} seconds")
            object_id = 0
            fps_int = int(fps) if fps > 0 else 30
            for sec in range(int(duration_seconds)): 
                ret,frame = cap.read()
                if not ret:
                    break

                frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
                timestamp = frame_no/fps if fps >0 else sec   
                
                results = yolo_model(frame)

                for result in results:
                    # annotated_img = result.plot()
                    for box in result.boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        padding = 20
                        height, width = frame.shape[:2]

                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(width, x2 + padding)
                        y2 = min(height, y2 + padding)

                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        crop_rgb = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
                        vec = embed_crop(crop_rgb,siglip_model,siglip_processor)
                        total_embd.append(vec)

                        metadata_rows.append({
                            "id":object_id,
                            "frame_no":frame_no,
                            "timestamp": round(timestamp,2),
                            "yolo_class": result.names[class_id],
                            "confidence": round(confidence,4)
                        })
                        object_id += 1

                processed_frame +=1
                for _ in range(fps_int - 1):
                    success, _ = cap.read()
                    if not success:
                        break
        else:
            print("Cannot open video")
    else:
        print("Unsupported Format.")
    print("Proccesed Frame:",processed_frame)
    total_embd= np.array(total_embd)
    print("embedding Shape :",total_embd.shape)

    metadata_df = pd.DataFrame(metadata_rows)
    print(metadata_df.head())


    np.save("data/video_embeddings/vid_embeddings.npy",total_embd)
    metadata_df.to_csv("data/meta_data/vid_metadata.csv",index = False)


finally:
    if cap is not None:
        cap.release()






