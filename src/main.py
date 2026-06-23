from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import shutil
import os

from src.video_processing import load_models, process_video
from src.vid_search_engine import video_search_engine
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
current_total_embd = None

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
    global current_video_path, current_metadata_df, current_total_embd

    # save uploaded video
    save_path = os.path.join("data/video_samples", video.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # process uploaded video
    metadata_df, total_embd = process_video(
        video_path=save_path,
        yolo_model=yolo_model,
        siglip_model=siglip_model,
        siglip_processor=siglip_processor,
        frame_sample_rate=1,
        padding=20,
        save_embeddings_path="data/video_embeddings/vid_embeddings.npy",
        save_metadata_path="data/meta_data/vid_metadata.csv"
    )

    # store current processed video data
    current_video_path = save_path
    current_metadata_df = metadata_df
    current_total_embd = total_embd

    return {
        "message": "Video uploaded and processed successfully",
        "video_path": save_path,
        "video_url": f"/videos/{video.filename}"
    }


class SearchRequest(BaseModel):
    query: str


@app.post("/search")
def search_video(request: SearchRequest):
    global current_video_path, current_metadata_df, current_total_embd

    if current_video_path is None or current_metadata_df is None or current_total_embd is None:
        raise HTTPException(
            status_code=400,
            detail="No video has been uploaded and processed yet."
        )

    query = request.query

    results = video_search_engine(
        query,
        current_metadata_df,
        current_total_embd,
        siglip_model,
        siglip_processor
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