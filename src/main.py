from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import shutil
import os
import time 

from src.vid_processing import load_models, process_video_with_tracking
from src.vid_search import video_search_engine
from src.thumbnail_creator import save_thumbnails

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# make sure folders exist
os.makedirs("data/video_samples", exist_ok=True)
os.makedirs("data/thumbnails", exist_ok=True)
os.makedirs("data/video_embeddings", exist_ok=True)
os.makedirs("data/meta_data", exist_ok=True)

# expose folders to frontend/browser
app.mount("/videos", StaticFiles(directory="data/video_samples"), name="videos")
app.mount("/thumbnails", StaticFiles(directory="data/thumbnails"), name="thumbnails")

# expose templates folder so script.js and style.css can be loaded
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# load models once
yolo_model, siglip_model, siglip_processor = load_models()

# global storage for currently uploaded/processed video
current_video_path = None
current_metadata_df = None
current_frame_emb = None
current_crop_emb = None

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-video")
def upload_video(video: UploadFile = File(...)):
    global current_video_path, current_metadata_df, current_crop_emb, current_frame_emb

    # save uploaded video
    save_path = os.path.join("data/video_samples", video.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)


    start_time = time.perf_counter()
    # process uploaded video
    metadata_df, crop_embeddings, frame_embeddings = process_video_with_tracking(
        video_path=save_path,
        yolo_model=yolo_model,
        siglip_model=siglip_model,
        siglip_processor=siglip_processor,
        sample_interval_sec=1,
        conf_threshold=0.5,
        padding=20,
        tracker_cfg="bytetrack.yaml",
        save_crop_emb_path="data/video_embeddings/crop_embeddings.npy",
        save_frame_emb_path="data/video_embeddings/frame_embeddings.npy",
        save_metadata_path="data/meta_data/vid_metadata.csv"

    )
    end_time = time.perf_counter()

    processing_time = end_time - start_time

    print(f"Video processing completed in {processing_time:.2f} seconds")

    # store current processed video data
    current_video_path = save_path
    current_metadata_df = metadata_df
    current_frame_emb = frame_embeddings
    current_crop_emb = crop_embeddings

    return {
        "message": "Video uploaded and processed successfully",
        "video_path": save_path,
        "video_url": f"/videos/{video.filename}"
    }


class SearchRequest(BaseModel):
    query: str


@app.post("/search")
def search_video(request: SearchRequest):
    global current_video_path, current_metadata_df, current_frame_emb,current_crop_emb

    if current_video_path is None or current_metadata_df is None:
        raise HTTPException(
            status_code=400,
            detail="No video has been uploaded and processed yet."
        )

    query = request.query

    results = video_search_engine(
        query,
        current_metadata_df,
        current_crop_emb,
        current_frame_emb,
        siglip_model,
        siglip_processor,
        top_k=5,
        threshold=None,
    )

    if results is None:
        return {
            "message": "No relevant match found in the video.",
            "results": []
        }

    thumbnails_data = save_thumbnails(current_video_path, results)

    # add thumbnail_url for frontend
    for item in thumbnails_data:
        filename = os.path.basename(item["thumbnail_path"])
        item["thumbnail_url"] = f"/thumbnails/{filename}"

    return {
        "message": "Search completed successfully",
        "results": thumbnails_data
    }