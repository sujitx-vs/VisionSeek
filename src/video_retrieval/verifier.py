import cv2
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()


def verify_frame(frame, query):
    """
    Verify whether a candidate frame satisfies the user's query.

    Returns:
        {
            "match": bool,
            "score": float
        }
    """

    # OpenCV (BGR) -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    prompt = f"""
You are a semantic retrieval verifier.

A retrieval system has returned this image for the following search query.

User Query:
"{query}"

Your task is to determine whether the image satisfies the user's search intent.

Consider:
- Correct object(s)
- Object attributes (color, clothing, size, etc.)
- Actions (walking, running, riding, sitting...)
- Scene context when relevant

Return ONLY valid JSON in exactly this format:

{{
    "match": true,
    "score": 0.93
}}

Rules:
- score must be between 0.0 and 1.0
- match=true only if the image clearly satisfies the query.
- Return no explanation.
- Return no markdown.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": frame_rgb
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[frame_rgb],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=0
        )

    response = processor.batch_decode(
        output[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    try:
        result = json.loads(response)

        result["score"] = float(result["score"])
        result["match"] = bool(result["match"])

        return result

    except Exception:
        return {
            "match": False,
            "score": 0.0
        }