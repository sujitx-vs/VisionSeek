import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel

siglip_model_name="google/siglip-base-patch16-224"

siglip_model = AutoModel.from_pretrained(siglip_model_name)
siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)

metadata = pd.read_csv("data/meta_data/vid_metadata.csv")
total_vector = np.load("data/video_embeddings/vid_embeddings.npy")

query = input("enter the query to search :")


def get_yolo_classes(query):
    query = query.lower()

    class_map = {
        "person": ["person", "man", "woman", "boy", "girl", "people", "walking", "standing"],
        "car": ["car", "sedan", "hatchback"],
        "bicycle": ["bicycle", "cycle", "bike"],
        "motorcycle": ["motorcycle", "motorbike"],
        "truck": ["truck", "lorry"],
        "bus": ["bus"],
        "van": ["van"]
    }
    matched_classes = []


    for yolo_class, keywords in class_map.items():
        for word in keywords:
            if word in query:
                matched_classes.append(yolo_class)
                break
    return matched_classes



inputs = siglip_processor(
    text=[query],
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    text_embeddings = siglip_model.get_text_features(**inputs)


query_vec = text_embeddings.squeeze(0).cpu().numpy()


matched_classes = get_yolo_classes(query)

if not matched_classes:
    raise SystemExit(
        "Query does not match any supported object class.\n"
        "Try searching for: person, car, bicycle, motorcycle, truck, bus, van"
    )
    
else:
    metadata_1 = metadata[metadata["yolo_class"].isin(matched_classes)].copy()

if len(metadata_1)==0:
    metadata_1 = metadata.copy()

index_1 = metadata_1.index.to_numpy()
vector_1 = total_vector[index_1]

scores = vector_1 @ query_vec

threshold = 0.2

if len(scores) == 0:
    print("No searchable objects found in the indexed video.")

else:
    top_k = min(5, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_k]

    best_score = scores[top_indices[0]]
    print("Best score:", best_score)

    if best_score < threshold:
        print("No relevant match found in the indexed video.")
    else:
        results = metadata_1.iloc[top_indices].copy()
        results["similarity"] = scores[top_indices]

        # keep only rows above threshold
        results = results[results["similarity"] >= threshold]

        if len(results) == 0:
            print("No results above the threshold.")
        else:
            print(results)
