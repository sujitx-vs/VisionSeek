import time

# Start the timer
start_time = time.perf_counter()


from video_processing import load_models, process_video
from vid_search_engine import video_search_engine

# load once
yolo_model, siglip_model, siglip_processor = load_models()

video_path = "data/video_samples/The CCTV People Demo 6.mp4"

metadata_df, total_embd = process_video(
    video_path=video_path,
    yolo_model=yolo_model,
    siglip_model=siglip_model,
    siglip_processor=siglip_processor,
    frame_sample_rate=1,
    padding=20,
    save_embeddings_path="data/video_embeddings/vid_embeddings.npy",
    save_metadata_path="data/meta_data/vid_metadata.csv"
)

print(metadata_df.head())
print(total_embd.shape)

#time consuming analysis
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

while True:
    query = input("enter the query to search :")
    results = video_search_engine(query)