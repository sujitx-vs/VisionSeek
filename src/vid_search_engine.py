import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel

siglip_model_name = "google/siglip-base-patch16-224"

siglip_model = AutoModel.from_pretrained(siglip_model_name)
siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)

metadata = pd.read_csv("data/meta_data/vid_metadata.csv")
total_vector = np.load("data/video_embeddings/vid_embeddings.npy")


def get_yolo_classes(query):
    query = query.lower()

    class_map = metadata["yolo_class"].dropna().astype(str).str.lower().unique()

    matched_classes = []

    for yolo_class in class_map:
        if yolo_class in query:
            matched_classes.append(yolo_class)

    return matched_classes


def video_search_engine(query, threshold=0.2, top_k=5):
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
        print("Query does not match any supported object class.")
        print("Try searching for: person, car, bicycle, motorcycle, truck, bus, van")
        return None

    else:
        metadata_1 = metadata[metadata["yolo_class"].isin(matched_classes)].copy()

    if len(metadata_1) == 0:
        metadata_1 = metadata.copy()

    index_1 = metadata_1.index.to_numpy()
    vector_1 = total_vector[index_1]

    scores = vector_1 @ query_vec

    if len(scores) == 0:
        print("No searchable objects found in the indexed video.")
        return None

    else:
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        best_score = scores[top_indices[0]]
        print("Best score:", best_score)

        if best_score < threshold:
            print("No relevant match found in the indexed video.")
            return None
        else:
            results = metadata_1.iloc[top_indices].copy()
            results["similarity"] = scores[top_indices]

            # keep only rows above threshold
            results = results[results["similarity"] >= threshold]

            if len(results) == 0:
                print("No results above the threshold.")
                return None
            else:
                print(results)
                return results