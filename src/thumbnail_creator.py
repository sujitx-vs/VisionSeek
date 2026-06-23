import os
import cv2


def save_thumbnails(video_path, result_df, output_folder="data/thumbnails"):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    thumbnail_data = []

    for i, (_, row) in enumerate(result_df.iterrows(), start=1):
        frame_no = int(row["frame_no"])
        timestamp = row["timestamp"]

        h, m, s = map(int, timestamp.split(":"))
        timestamp_seconds = h * 3600 + m * 60 + s

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if ret:
            thumbnail_path = os.path.join(output_folder, f"thumb_{i}.jpg")
            cv2.imwrite(thumbnail_path, frame)

            thumbnail_data.append({
                "thumbnail_path": thumbnail_path, 
                "timestamp": timestamp,
                "timestamp_seconds": timestamp_seconds,
                "frame_no": frame_no
            })

            print(f"Saved: {thumbnail_path} -> {timestamp}")

    cap.release()
    return thumbnail_data

