# 🎥 VisionSeek

**AI-Powered Semantic Video Search for Surveillance Footage**

VisionSeek is a Computer Vision and Multimodal AI project that enables users to search long surveillance videos using natural language instead of manually reviewing hours of footage.

Users can upload a video, process it once, and retrieve relevant moments by simply typing queries such as:

* `"a person"`
* `"white car"`
* `"person carrying a backpack"`
* `"truck parked near the road"`
* `"skateboard"`

The system combines object detection, multi-object tracking, vision-language embeddings, and multimodal verification to accurately locate relevant video segments.

---

# 📖 Overview

Traditional CCTV analysis requires manually scrubbing through hours of recordings.

VisionSeek converts a video into a searchable semantic index by combining:

* YOLO object detection
* ByteTrack multi-object tracking
* SigLIP2 vision-language embeddings
* Qwen2.5-VL semantic verification

Instead of searching by filename or timestamp, users search by meaning.

---

# ✨ Features

* 🎥 Upload any surveillance video
* 🚀 Automatic video indexing
* 👥 Multi-object tracking using ByteTrack
* 🧠 Semantic search using SigLIP2 embeddings
* 🤖 Qwen2.5-VL verification to reduce false positives
* 🖼 Thumbnail generation for matched results
* ▶ Click thumbnail to jump directly to the corresponding timestamp
* 💻 Interactive Gradio interface

---

# 🏗 Current Pipeline

```
                 Upload Video
                      │
                      ▼
              OpenCV Video Loader
                      │
                      ▼
        YOLO12 Object Detection + ByteTrack
                      │
                      ▼
         Multi-object Track Generation
                      │
                      ▼
    Store:
       • Track ID
       • Start Time
       • End Time
       • Best 5 Frames
                      │
                      ▼
        Generate SigLIP2 Embeddings
         ├── Full Frame
         └── Object Crop
                      │
                      ▼
        Metadata + Embedding Index
                      │
                      ▼
            User Text Query
                      │
                      ▼
      SigLIP2 Text Embedding
                      │
                      ▼
        Semantic Similarity Search
         ├── Crop Search
         └── Frame Search
                      │
                      ▼
          Merge Top Candidates
                      │
                      ▼
      Qwen2.5-VL Verification
                      │
                      ▼
       Generate Result Thumbnails
                      │
                      ▼
      Click Thumbnail → Jump to Video
```

---

# 🧠 Models Used

## Object Detection

* YOLO12x

Used for detecting objects in every frame.

---

## Multi-Object Tracking

* ByteTrack

Tracks detected objects across the video and assigns persistent Track IDs.

For every tracked object, VisionSeek stores:

* first appearance
* last appearance
* best 5 confidence frames

---

## Vision-Language Embedding

* SigLIP2 SO400M Patch14

Used to generate embeddings for

* full frames
* detected object crops

These embeddings are used for semantic retrieval.

---

## Retrieval Verification

* Qwen2.5-VL-3B-Instruct

After semantic retrieval, the top candidates are verified using Qwen Vision-Language Model to reduce false positives before presenting results.

---

# 🚀 Current Project Status

## ✅ Completed

* Video upload interface
* Video preprocessing pipeline
* YOLO object detection
* ByteTrack multi-object tracking
* First/Last appearance tracking
* Best-frame selection
* SigLIP2 frame embeddings
* SigLIP2 crop embeddings
* Semantic retrieval
* Qwen-based result verification
* Thumbnail generation
* Interactive Gradio UI
* GPU support (CUDA)

---

## 🚧 Current Limitations

This is still a prototype and has several known limitations:

* Processes one video at a time.
* Embeddings are stored in memory (FAISS integration pending).
* Long videos require significant processing time.
* Retrieval accuracy depends heavily on YOLO detections.
* Similar-looking objects may occasionally produce false positives.

---

# 📂 Project Structure

```
VisionSeek/
│
├── app/
│   └── app.py                 # Gradio Interface
│
├── src/
│   ├── video_processing/
│   │      video_open.py
│   │      video_track.py
│   │      video_embedding.py
│   │
│   ├── video_retrieval/
│   │      semantic_search.py
│   │      verifier.py
│   │      thumbnail_fetch.py
│   │      text_embedding.py
│   │
│   └── utils/
│          device.py
│
├── models/
│
├── data/
│      thumbnails/
│
├── notebooks/
│      Research experiments
│
├── requirements.txt
│
└── README.md
```

---

# 🛠 Technologies

* Python
* OpenCV
* PyTorch
* Transformers
* Ultralytics YOLO12
* ByteTrack
* SigLIP2
* Qwen2.5-VL
* NumPy
* Pandas
* Gradio

---

# 🔮 Roadmap

The current version is a functional prototype. Future work includes:

* FAISS vector database integration
* Multiple video indexing
* Persistent embedding storage
* Faster approximate nearest-neighbor search
* Temporal action retrieval
* Scene-level indexing
* Better ranking strategy
* Distributed processing for long videos

---

# 🎯 Future Vision

VisionSeek aims to evolve into a production-ready semantic video search engine capable of indexing thousands of hours of surveillance footage.

The long-term goal is to enable users to retrieve complex events using natural language rather than manually reviewing video recordings.

---

# 👨‍💻 Author

**Sujith V S**

Computer Vision • Multimodal AI • Video Retrieval

---

# 📄 License

This project is intended for research and educational purposes.
