import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
from pprint import pprint
from transformers import AutoProcessor, AutoModel
from ultralytics import YOLO
import pandas as pd

from video_open import vid_open
from video_track import vid_track
from video_embedding import vid_embd
from video_retrieval.text_embedding import embed_text_query
from video_retrieval.semantic_search import semantic_search




yolo_model_path = "models/yolo12x.pt" 
yolo_model = YOLO(yolo_model_path)


siglip_model_name="google/siglip2-so400m-patch14-224"
siglip_model = AutoModel.from_pretrained(siglip_model_name)
siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)

VIDEO_PATH = r"data\video_samples\CCTV Kit sample Video 480P.mp4"


def main():
    start_time = time.perf_counter()

    # Open video once
    cap = vid_open(VIDEO_PATH)

    # Tracking
    tracked_objects = vid_track(cap,yolo_model)

    # Embedding
    frame_embeddings, crop_embeddings, metadata_df = vid_embd(tracked_objects, cap,siglip_model,siglip_processor)

    # Done with video
    cap.release()

    end_time = time.perf_counter()

    print("\n========== TRACKED OBJECTS ==========\n")
    pprint(tracked_objects, sort_dicts=False)

    print("\n========== EMBEDDING ==========")
    print("Frame Embeddings Shape :", frame_embeddings.shape)
    print("Crop Embeddings Shape  :", crop_embeddings.shape)
    print("Metadata Shape         :", metadata_df.shape)

    print("\n========== SUMMARY ==========")
    print(f"Total Tracks: {len(tracked_objects)}")
    print(f"Execution Time: {end_time - start_time:.3f} seconds")

    query = input("Enter the query to search")

    fetch_start_time = time.perf_counter()
    query_embedding = embed_text_query(query,siglip_model,siglip_processor)

    crop_results,frame_results = semantic_search(query_embedding,crop_embeddings,frame_embeddings,metadata_df)

    fetch_end_time = time.perf_counter()

    print("\n========== CROP RESULTS ==========\n")
    print(crop_results)
    print("\n========== FRAME RESULTS ==========\n")
    print(frame_results)
    print(f"Execution Time: {fetch_end_time - fetch_start_time:.3f} seconds")

    # merging the results
    final_results = pd.concat([crop_results,frame_results],ignore_index=True)

    THRESHOLD = 0.045

    # filtering using threshold
    final_results = final_results[final_results["score"] >= THRESHOLD].copy()

    final_results = final_results.drop_duplicates(subset="track_id",keep="first").reset_index(drop=True)

    final_results = final_results.sort_values(by="score",ascending=False).reset_index(drop=True)
    print(final_results)   






if __name__ == "__main__":
    main()