import os
import cv2


def save_thumbnails(video_path, result_df, output_folder="data/thumbnails"):
    """
    Save thumbnails for all verified retrieval results.

    Parameters
    ----------
    video_path : str
    result_df : pd.DataFrame
    output_folder : str

    Returns
    -------
    list[dict]
        [
            {
                "thumbnail_path": ...,
                "frame_no": ...,
                "timestamp": ...,
                "track_id": ...,
                "score": ...
            }
        ]
    """

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    thumbnail_data = []

    for i, (_, row) in enumerate(result_df.iterrows(), start=1):

        frame_no = int(row["frame_no"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret:
            continue

        thumbnail_path = os.path.join(output_folder, f"thumb_{i}.jpg")

        cv2.imwrite(thumbnail_path, frame)

        thumbnail_data.append({
            "thumbnail_path": thumbnail_path,
            "track_id": row["track_id"],
            "frame_no": frame_no,
            "timestamp": row["timestamp"],
            "score": row.get("verifier_score", row["score"])
        })

        print(f"Saved: {thumbnail_path}")

    cap.release()

    return thumbnail_data