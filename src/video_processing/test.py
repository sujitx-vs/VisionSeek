import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from pprint import pprint

from transformers import AutoProcessor, AutoModel
from ultralytics import YOLO

from video_open import vid_open
from video_track import vid_track
from video_embedding import vid_embd
from video_retrieval.text_embedding import embed_text_query
from video_retrieval.semantic_search import semantic_search
from video_retrieval.frame_fetch import fetch_frame
from video_retrieval.verifier import verify_frame


# -------------------------------
# Models
# -------------------------------

yolo_model_path = "models/yolo12x.pt"
yolo_model = YOLO(yolo_model_path)

siglip_model_name = "google/siglip2-so400m-patch14-224"
siglip_model = AutoModel.from_pretrained(siglip_model_name)
siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)

VIDEO_PATH = r"data\video_samples\CCTV Kit sample Video 480P.mp4"

# -------------------------------
# Cache
# -------------------------------

CACHE_DIR = "cache"

FRAME_EMB_PATH = os.path.join(CACHE_DIR, "frame_embeddings.npy")
CROP_EMB_PATH = os.path.join(CACHE_DIR, "crop_embeddings.npy")
META_PATH = os.path.join(CACHE_DIR, "metadata.pkl")
TRACK_PATH = os.path.join(CACHE_DIR, "tracked_objects.pkl")

os.makedirs(CACHE_DIR, exist_ok=True)


def main():

    # ---------------------------------------------------
    # Load cache if available
    # ---------------------------------------------------

    if (
        os.path.exists(FRAME_EMB_PATH)
        and os.path.exists(CROP_EMB_PATH)
        and os.path.exists(META_PATH)
        and os.path.exists(TRACK_PATH)
    ):

        print("\n========== LOADING CACHE ==========\n")

        frame_embeddings = np.load(FRAME_EMB_PATH)
        crop_embeddings = np.load(CROP_EMB_PATH)
        metadata_df = pd.read_pickle(META_PATH)

        with open(TRACK_PATH, "rb") as f:
            tracked_objects = pickle.load(f)

    else:

        print("\n========== BUILDING CACHE ==========\n")

        start_time = time.perf_counter()

        cap = vid_open(VIDEO_PATH)

        tracked_objects = vid_track(cap, yolo_model)

        frame_embeddings, crop_embeddings, metadata_df = vid_embd(
            tracked_objects,
            cap,
            siglip_model,
            siglip_processor
        )

        cap.release()

        end_time = time.perf_counter()

        np.save(FRAME_EMB_PATH, frame_embeddings)
        np.save(CROP_EMB_PATH, crop_embeddings)

        metadata_df.to_pickle(META_PATH)

        with open(TRACK_PATH, "wb") as f:
            pickle.dump(tracked_objects, f)

        print("\n========== CACHE SAVED ==========\n")

        print(f"Execution Time : {end_time-start_time:.2f} sec")

    print("\n========== EMBEDDINGS ==========")
    print("Frame :", frame_embeddings.shape)
    print("Crop  :", crop_embeddings.shape)
    print("Meta  :", metadata_df.shape)

    print(f"\nTracks : {len(tracked_objects)}")

    # ---------------------------------------------------
    # Retrieval
    # ---------------------------------------------------

    query = input("\nEnter Query : ")

    start = time.perf_counter()

    query_embedding = embed_text_query(
        query,
        siglip_model,
        siglip_processor
    )

    crop_results, frame_results = semantic_search(
        query_embedding,
        crop_embeddings,
        frame_embeddings,
        metadata_df
    )

    end = time.perf_counter()

    print("\n========== Crop Results ==========\n")
    print(crop_results)

    print("\n========== Frame Results ==========\n")
    print(frame_results)

    print(f"\nRetrieval Time : {end-start:.3f} sec")

    # ---------------------------------------------------
    # Merge
    # ---------------------------------------------------

    final_results = pd.concat(
        [crop_results, frame_results],
        ignore_index=True
    )

    THRESHOLD = 0.045

    final_results = (
        final_results[final_results["score"] >= THRESHOLD]
        .drop_duplicates(subset="track_id", keep="first")
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    print("\n========== Final Results ==========\n")
    print(final_results)

    # ---------------------------------------------------
    # Verification
    # ---------------------------------------------------

    cap = vid_open(VIDEO_PATH)

    for _, row in final_results.iterrows():

        frame = fetch_frame(
            cap,
            int(row["frame_no"])
        )

        if frame is None:
            continue

        result = verify_frame(frame, query)

        print("\nFrame :", row["frame_no"])
        print(result)

    cap.release()


if __name__ == "__main__":
    main()
    