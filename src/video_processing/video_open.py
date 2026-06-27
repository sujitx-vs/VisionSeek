import cv2

VALID_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
def vid_open(video_path):
    if not video_path.lower().endswith(VALID_VIDEO_EXTS):
        raise ValueError(f"Unsupported video format: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    else:
        return cap