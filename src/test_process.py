from vid_processing import load_models, process_video_with_tracking

video_path = "data/video_samples/The CCTV People Demo 6.mp4"

yolo_model, siglip_model, siglip_processor = load_models()

metadata_df, crop_embeddings, frame_embeddings = process_video_with_tracking(
    video_path=video_path,
    yolo_model=yolo_model,
    siglip_model=siglip_model,
    siglip_processor=siglip_processor,
    sample_interval_sec=0.5,
    conf_threshold=0.5,
    padding=20,
    tracker_cfg="bytetrack.yaml",
    save_crop_emb_path="data/video_embeddings/crop_embeddings.npy",
    save_frame_emb_path="data/video_embeddings/frame_embeddings.npy",
    save_metadata_path="data/meta_data/vid_metadata.csv"
)

print(metadata_df.head())
print("Crop embeddings:", crop_embeddings.shape)
print("Frame embeddings:", frame_embeddings.shape)