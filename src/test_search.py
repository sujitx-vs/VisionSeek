import numpy as np
import pandas as pd
from vid_processing import load_models
from vid_search import video_search_engine

# load models
yolo_model, siglip_model, siglip_processor = load_models()

# load saved outputs from video processing
metadata_df = pd.read_csv("data/meta_data/vid_metadata.csv")
crop_embeddings = np.load("data/video_embeddings/crop_embeddings.npy")

# query
query = input("Enter search query: ")

results = video_search_engine(
    query=query,
    metadata_df=metadata_df,
    crop_embeddings=crop_embeddings,
    siglip_model=siglip_model,
    siglip_processor=siglip_processor,
    top_k=5,
    threshold=None   # keep None for now, tune later
)

if results is None:
    print("No results found.")
else:
    print("\nTop matches:")
    print(results)