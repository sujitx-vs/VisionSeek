"""
Gradio front-end for the video semantic search pipeline.

Pipeline:
  Upload video -> shows in player
  Click "Process Video" -> YOLO -> Tracking -> Embedding -> index ready
  Search box enabled -> query -> semantic search -> merge -> verify -> thumbnails
  Click thumbnail -> tiny JS seeks the <video> element to that timestamp

Run with:
    python app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import time
import traceback

import cv2
import gradio as gr
import pandas as pd

if os.environ.get("SMOKE_TEST"):
    class YOLO:
        def __init__(self, path): pass
    class AutoModel:
        @staticmethod
        def from_pretrained(name): return object()
    class AutoProcessor:
        @staticmethod
        def from_pretrained(name): return object()
else:
    from transformers import AutoProcessor, AutoModel
    from ultralytics import YOLO

from src.video_processing.video_open import vid_open
from src.video_processing.video_track import vid_track
from src.video_processing.video_embedding import vid_embd
from src.video_retrieval.text_embedding import embed_text_query
from src.video_retrieval.semantic_search import semantic_search
from src.video_retrieval.verifier import verify_frame
from src.video_retrieval.thumbnail_fetch import save_thumbnails

# -------------------------------------------------------------------
# Models (loaded once, shared across all sessions/requests)
# -------------------------------------------------------------------

print("Loading YOLO model...")
yolo_model = YOLO("models/yolo12x.pt")

print("Loading SigLIP model...")
SIGLIP_NAME = "google/siglip2-so400m-patch14-224"
siglip_model = AutoModel.from_pretrained(SIGLIP_NAME)
siglip_processor = AutoProcessor.from_pretrained(SIGLIP_NAME)

THUMB_DIR = "thumbnails"
os.makedirs(THUMB_DIR, exist_ok=True)

SCORE_THRESHOLD = 0.045


# -------------------------------------------------------------------
# Stage 1: Process Video  (YOLO -> Tracking -> Embedding -> Index)
# -------------------------------------------------------------------

def process_video(video_path, progress=gr.Progress(track_tqdm=True)):
    """
    Runs the full indexing pipeline on the uploaded video and stashes
    everything the search stage needs inside gr.State.
    """
    if not video_path:
        raise gr.Error("Please upload a video first.")

    try:
        progress(0.05, desc="Opening video...")
        cap = vid_open(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        progress(0.15, desc="Running YOLO + tracking...")
        tracked_objects = vid_track(cap, yolo_model)

        progress(0.55, desc="Generating embeddings...")
        frame_embeddings, crop_embeddings, metadata_df = vid_embd(
            tracked_objects, cap, siglip_model, siglip_processor
        )

        cap.release()

        progress(1.0, desc="Index ready.")

        pipeline_state = {
            "video_path": video_path,
            "fps": fps,
            "tracked_objects": tracked_objects,
            "frame_embeddings": frame_embeddings,
            "crop_embeddings": crop_embeddings,
            "metadata_df": metadata_df,
        }

        status = (
            f"✅ Indexed {len(tracked_objects)} tracks · "
            f"{frame_embeddings.shape[0]} frame embeddings · "
            f"{crop_embeddings.shape[0]} crop embeddings.\n"
            f"Search is now enabled."
        )

        return (
            pipeline_state,                      # state
            status,                              # status textbox
            gr.update(interactive=True),          # search box
            gr.update(interactive=True),          # search button
        )

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Processing failed: {e}")


# -------------------------------------------------------------------
# Stage 2: Search  (semantic search -> merge -> verify -> thumbnails)
# -------------------------------------------------------------------

def run_search(query, pipeline_state, progress=gr.Progress(track_tqdm=True)):
    if not pipeline_state:
        raise gr.Error("Process the video before searching.")
    if not query or not query.strip():
        raise gr.Error("Enter a search query.")

    try:
        video_path = pipeline_state["video_path"]
        fps = pipeline_state["fps"]
        frame_embeddings = pipeline_state["frame_embeddings"]
        crop_embeddings = pipeline_state["crop_embeddings"]
        metadata_df = pipeline_state["metadata_df"]

        progress(0.1, desc="Embedding query...")
        query_embedding = embed_text_query(query, siglip_model, siglip_processor)

        progress(0.3, desc="Running semantic search...")
        crop_results, frame_results = semantic_search(
            query_embedding, crop_embeddings, frame_embeddings, metadata_df
        )

        progress(0.5, desc="Merging results...")
        final_results = pd.concat([crop_results, frame_results], ignore_index=True)
        final_results = (
            final_results[final_results["score"] >= SCORE_THRESHOLD]
            .drop_duplicates(subset="track_id", keep="first")
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        if final_results.empty:
            return [], "No matches found above the score threshold.", []

        progress(0.65, desc="Verifying candidates...")
        cap = vid_open(video_path)
        verified_rows = []

        for _, row in final_results.iterrows():
            result = verify_frame(cap, int(row["frame_no"]), fps, query)
            if result["match"]:
                verified_row = row.copy()
                verified_row["frame_no"] = result["frame_no"]
                verified_row["timestamp"] = result["timestamp"]
                verified_row["verifier_score"] = result["score"]
                verified_rows.append(verified_row)

        cap.release()

        verified_df = pd.DataFrame(verified_rows)

        if verified_df.empty:
            return [], "Verifier rejected all candidates.", []

        progress(0.85, desc="Generating thumbnails...")
        thumbnails = save_thumbnails(video_path, verified_df)
        # thumbnails is assumed to be a list of image filepaths, aligned
        # row-for-row with verified_df. Adjust the zip below if your
        # save_thumbnails returns (path, timestamp) tuples already.

        gallery_items = []
        timestamps = []
        for path, (_, row) in zip(thumbnails, verified_df.iterrows()):
            ts = float(row["timestamp"])
            caption = f"t={ts:.2f}s · score={row.get('verifier_score', row.get('score', 0)):.3f}"
            gallery_items.append((path, caption))
            timestamps.append(ts)

        progress(1.0, desc="Done.")

        status = f"Found {len(gallery_items)} verified result(s)."
        return gallery_items, status, timestamps

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Search failed: {e}")


# -------------------------------------------------------------------
# Stage 3: Thumbnail click -> seek video player
# -------------------------------------------------------------------

def on_thumbnail_select(evt: gr.SelectData, timestamps):
    """
    Reads which gallery item was clicked and returns its timestamp.
    The actual seeking happens in the JS below, which reads this value.
    """
    if timestamps is None or evt.index >= len(timestamps):
        return 0.0
    return float(timestamps[evt.index])


SEEK_JS = """
(seconds) => {
    const video = document.querySelector("#video-player video");
    if (video) {
        video.currentTime = seconds;
        video.play();
    }
    return seconds;
}
"""


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

with gr.Blocks(title="Video Semantic Search") as demo:
    gr.Markdown("## 🔍 Video Semantic Search\nUpload a video, process it, then search for anything that happened in it.")

    pipeline_state = gr.State(None)        # holds embeddings/tracks/etc. per session
    timestamps_state = gr.State([])        # holds the timestamp for each gallery thumbnail
    seek_target = gr.Number(visible=False) # bridges Python -> JS for seeking

    with gr.Row():
        with gr.Column(scale=2):
            video_player = gr.Video(label="Video", elem_id="video-player")
            process_btn = gr.Button("Process Video", variant="primary")
            process_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            search_box = gr.Textbox(
                label="Search query",
                placeholder="Process the video first...",
                interactive=False,
            )
            search_btn = gr.Button("Search", interactive=False)
            search_status = gr.Textbox(label="Search status", interactive=False)
            gallery = gr.Gallery(
                label="Results (click a thumbnail to jump to that moment)",
                columns=4,
                height=400,
            )

    # 1) Upload -> immediately show in player (gr.Video does this natively
    #    via its `value`/input binding, no extra wiring needed)

    # 2) Process Video -> run pipeline -> enable search
    process_btn.click(
        fn=process_video,
        inputs=[video_player],
        outputs=[pipeline_state, process_status, search_box, search_btn],
    )

    # 3) Search -> semantic search -> merge -> verify -> thumbnails
    search_btn.click(
        fn=run_search,
        inputs=[search_box, pipeline_state],
        outputs=[gallery, search_status, timestamps_state],
    )
    search_box.submit(
        fn=run_search,
        inputs=[search_box, pipeline_state],
        outputs=[gallery, search_status, timestamps_state],
    )

    # 4) Thumbnail click -> compute timestamp -> small JS seeks the player
    gallery.select(
        fn=on_thumbnail_select,
        inputs=[timestamps_state],
        outputs=[seek_target],
    ).then(
        fn=None,
        inputs=[seek_target],
        outputs=[seek_target],
        js=SEEK_JS,
    )


if __name__ == "__main__":
    demo.queue().launch()
