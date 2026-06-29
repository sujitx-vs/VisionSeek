import cv2
import json
import torch
import transformers
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

print("Transformers:", transformers.__version__)

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"





print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

try:
    print("Loading model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None
    )

    print("Loaded successfully.")

except Exception as e:
    import traceback
    traceback.print_exc()

model.eval()

device = torch.device("cpu")
model.to(device)

print("Model loaded!")
print("Device:", device)


def verify_frame(frame, query):
    """
    Verify whether a candidate frame satisfies the query.

    Returns
    -------
    {
        "match": bool,
        "score": float
    }
    """

    print("\n==============================")
    print("Starting verification...")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    prompt = f"""
You are a semantic retrieval verifier.

User Query:
"{query}"

Look at the image.

Return ONLY valid JSON.

Example:

{{
    "match": true,
    "score": 0.93
}}

Rules:
- score must be between 0 and 1
- no markdown
- no explanation
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

    print("Applying chat template...")

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Preparing inputs...")

    inputs = processor(
        text=[text],
        images=[frame_rgb],
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Calling generate...")

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )

    print("Generation finished.")

    response = processor.batch_decode(
        output[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

    print("\nRAW RESPONSE:")
    print("--------------------------------")
    print(repr(response))
    print("--------------------------------")

    try:

        result = json.loads(response)

        result["match"] = bool(result["match"])
        result["score"] = float(result["score"])

        print("JSON Parsed Successfully.")

        return result

    except Exception as e:

        print("\nJSON Parsing Failed")
        print(e)

        return {
            "match": False,
            "score": 0.0,
        }