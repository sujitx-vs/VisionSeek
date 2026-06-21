from video_processing import load_models, process_video

# load once
yolo_model, siglip_model, siglip_processor = load_models()

video_path = "data/video_samples/CCTV Kit sample Video 480P.mp4"

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