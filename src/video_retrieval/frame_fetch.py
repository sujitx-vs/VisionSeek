import cv2
def fetch_frame(cap, frame_no):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ret, frame = cap.read()

    if not ret:
        return None

    return frame