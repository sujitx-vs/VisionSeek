import os 
import cv2

from ultralytics import YOLO

VALID_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def vid_track(cap,yolo_model,device):
   
    

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



    if fps <= 0:
        fps = 30.0

    duration_seconds = total_frames / fps

    print("----- VIDEO INFO -----")
    # print(f"Path: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration_seconds:.2f} sec")

    tracked_objects = {}
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
                break
        
        results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        device=device
        )
        
        timestamp = frame_no / fps

        if not results:
                frame_no +=1
                continue
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            frame_no += 1
            continue

        boxes = result.boxes
        for box in boxes:
            if box.id is None:
                continue
            track_id = int(box.id.item()) 
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # print(
            #     f"Frame {frame_no}"
            #     f"\nTrack {track_id}"
            #     f"\nClass {yolo_model.names[cls]}"
            #     f"\nConf {conf:.2f}"
            #     f"\nBBox {[int(x1), int(y1), int(x2), int(y2)]}"
            # )

            if track_id not in tracked_objects:

                tracked_objects[track_id] = {
                
                    "track_id": track_id,

                    "class_id": cls,
                    "class_name": yolo_model.names[cls],

                    "start_frame": frame_no,
                    "last_frame": frame_no,

                    "start_time": timestamp,
                    "last_time": timestamp,

                    "best_frames": []
                }
            tracked_objects[track_id]["last_frame"] = frame_no
            tracked_objects[track_id]["last_time"] = timestamp

            frame_data = {
                "frame_no": frame_no,
                "timestamp": timestamp,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            }

            best_frames = tracked_objects[track_id]["best_frames"]

            if len(best_frames) < 5:
                best_frames.append(frame_data)

            else:
                # Find the stored frame with the lowest confidence
                min_index = min(
                    range(len(best_frames)),
                    key=lambda i: best_frames[i]["confidence"]
                )

                # Replace it if the current frame has higher confidence
                if conf > best_frames[min_index]["confidence"]:
                    best_frames[min_index] = frame_data


        frame_no += 1
    return tracked_objects
