import cv2
import json
import torch
import transformers
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from src.utils.device import get_device

device = get_device()

print("Transformers:", transformers.__version__)

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading model...")

model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
    )
    .to(device)
)

model.eval()

print("Model loaded!")
print("Device:", device)


def _verify_single_frame(frame, query):
    """
    Verify a single frame using Qwen.
    """

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    prompt = f"""
You are a semantic retrieval verifier.

User Query:
"{query}"

Look at the image carefully.

Return ONLY valid JSON.

Example:

{{
    "match": true,
    "score": 0.93
}}

Rules:
- score must be between 0 and 1
- match=true only if the frame clearly satisfies the query
- no explanation
- no markdown
- JSON only
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": frame_rgb,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[frame_rgb],
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )

    response = processor.batch_decode(
        output[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    print("\nRAW RESPONSE:")
    print(response)

    try:

        result = json.loads(response)

        return {
            "match": bool(result["match"]),
            "score": float(result["score"]),
        }

    except Exception:

        return {
            "match": False,
            "score": 0.0,
        }


def verify_frame(
    cap,
    tracked_objects,
    track_id,
    start_frame,
    query,
    max_frames=30,
):
    """
    Verify only frames belonging to the retrieved object.

    Parameters
    ----------
    cap : cv2.VideoCapture
    tracked_objects : dict
    track_id : int
    start_frame : int
    query : str
    max_frames : int

    Returns
    -------
    {
        "match": bool,
        "score": float,
        "frame_no": int | None,
        "timestamp": float | None
    }
    """

    if track_id not in tracked_objects:
        return {
            "match": False,
            "score": 0.0,
            "frame_no": None,
            "timestamp": None,
        }

    obj = tracked_objects[track_id]

    candidate_frames = []

# First appearance
    candidate_frames.append(
        (
            obj["start_frame"],
            obj["start_time"]
        )
    )
    
    # Best frames
    for frame in obj["best_frames"]:
        candidate_frames.append(
            (
                frame["frame_no"],
                frame["timestamp"]
            )
        )
    
    # Remove duplicates
    candidate_frames = list(dict.fromkeys(candidate_frames))
    
    # Check first appearance + best frames
    for frame_no, timestamp in candidate_frames:
    
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    
        ret, frame = cap.read()
    
        if not ret:
            continue
        
        print(f"Checking candidate frame {frame_no}")
    
        result = _verify_single_frame(frame, query)
    
        if result["match"]:
        
            return {
                "match": True,
                "score": result["score"],
                "frame_no": frame_no,
                "timestamp": timestamp,
            }
    
    # If none matched, search through the object's lifetime
    checked = 0
    
    for frame_no in range(obj["start_frame"], obj["last_frame"] + 1):
    
        if frame_no < start_frame:
            continue
        
        if checked >= max_frames:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    
        ret, frame = cap.read()
    
        if not ret:
            break
        
        print(f"Checking lifetime frame {frame_no}")
    
        result = _verify_single_frame(frame, query)
    
        checked += 1
    
        if result["match"]:
        
            fps = cap.get(cv2.CAP_PROP_FPS)
    
            return {
                "match": True,
                "score": result["score"],
                "frame_no": frame_no,
                "timestamp": frame_no / fps,
            }

    return {
        "match": False,
        "score": 0.0,
        "frame_no": None,
        "timestamp": None,
    }