🎥 VisionSeek
AI-Powered Video Search Engine for CCTV and Surveillance Footage

📖 Overview
VisionSeek is an intelligent video retrieval system designed to help users quickly locate objects and events within long surveillance videos using natural language queries.

Instead of manually reviewing hours of CCTV footage, users can search for specific objects or scenes and instantly retrieve the relevant timestamps where they appear.

Example queries:

"truck"

"person wearing a backpack"

"red car"

"man carrying a box"

The system aims to provide a faster and more efficient way to analyze surveillance footage using modern Computer Vision and Multimodal AI techniques.

🎯 Problem Statement
Reviewing CCTV recordings is a time-consuming process, especially when searching for a specific object or event within hours of footage.

VisionSeek aims to solve this problem by enabling semantic search over video content, allowing users to find relevant moments using simple text descriptions.

🚀 Current Development Status
This project is currently under active development. While the repository contains early-stage research experiments, a raw, functional prototype has been successfully built and tested. The baseline pipeline is working, though optimizing detection depth and retrieval accuracy remains the primary focus moving forward.

Completed
Video Processing Pipeline: Seamless frame extraction and preprocessing.

YOLO-Based Detection: Automated object detection bounding boxes per frame.

Cropping & Embedding Generation: Dynamic cropping of YOLO bounding boxes passed directly into a SigLIP model to extract high-quality vision-language embeddings.

Vector Indexing: Storage of generated embeddings mapped alongside exact timestamps.

Search & Retrieval Mechanism: Takes a user's natural language input, matches it against YOLO labels, calculates cosine similarity on SigLIP embeddings, and successfully returns matched thumbnails mapped to their original video timestamps.

Model Evaluation: Initial comparative experiments between CLIP and SigLIP models.

In Progress & Current Challenges
While the initial prototype serves as a functional proof of concept, it faces a few key technical limitations that are actively being addressed:

Single Object Bottleneck: The current pipeline frequently defaults to index/detect only one dominant object per frame, occasionally missing secondary objects even if they are clearly visible in the video.

Accuracy Thresholds: Fine-grained semantic matching (e.g., distinguishing specific clothing attributes or subtle actions) requires further optimization.

Vector Database Migration: Transitioning the pipeline into a structured FAISS vector database for faster indexing.

Planned Features
Multi-object tracking and indexing per frame to eliminate the single-object bottleneck.

Advanced timestamp-based UI navigation.

Interactive Streamlit web application dashboard.

Batch video uploading, parallel indexing, and real-time semantic search results.

🏗 Pipeline Architecture
Plaintext
       [ Raw Video Input ]
                │
                ▼
       [ Frame Extraction ]
                │
                ▼
     [ YOLO Object Detection ]
                │
                ▼
       [ Object Cropping ] ─── (Extracts bounding box regions)
                │
                ▼
     [ SigLIP Embeddings ] ─── (Generates multi-modal feature vectors)
                │
                ▼
   [ Vector & Timestamp Map ] ─── (Saves embedding linked to exact timestamp)
                │
                ▼
   [ User Text Query Match ] ─── (Compares YOLO labels + Cosine Similarity)
                │
                ▼
[ Matched Thumbnail + Timestamp ]
🛠 Technologies
Python

OpenCV

YOLOv8

SigLIP (Hugging Face Transformers)

Cosine Similarity / SciPy

FAISS (Integration Pending)

NumPy & Pandas

Streamlit (UI Pending)

📂 Project Structure
Plaintext
VisionSeek
│
├── data            # Raw surveillance video samples
├── notebooks       # CLIP vs SigLIP research and initial testing
├── src             # Core processing, YOLO cropping, and embedding script pipelines
├── app             # Streamlit application UI layout (In Progress)
├── models          # Local weights for YOLO and model configs
├── requirements.txt
└── README.md
🔮 Closing Statement & Future Goals
VisionSeek was born out of a desire to explore the intersection of Computer Vision, Multimodal AI, and rapid Video Retrieval Systems.

The first milestone—proving that a text query can successfully isolate an object crop and map it back to a timestamped thumbnail—has been achieved. The journey ahead is focused entirely on breaking past the single-object detection limit, refining the retrieval confidence scores, and scaling the system to handle complex, multi-object crowded surveillance environments.

📌 Project Status
🚧 Work in Progress (Functional Prototype Achieved)

👨‍💻 Author
Sujith V S