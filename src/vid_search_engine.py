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

inputs = siglip_processor(
    text=[query],
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    text_embeddings = siglip_model.get_text_features(**inputs)

# L2 Normalization
text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

query_vec = text_embeddings.squeeze(0).cpu().numpy()

scores = total_vector @ query_vec

top_k = 5
top_indices = np.argsort(scores)[::-1][:top_k]

results = metadata.iloc[top_indices].copy()
results["similarity"] = scores[top_indices]
print(results)