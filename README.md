# 🎥 VisionSeek

> AI-Powered Video Search Engine for CCTV and Surveillance Footage

## 📖 Overview

VisionSeek is an intelligent video retrieval system designed to help users quickly locate objects and events within long surveillance videos using natural language queries.

Instead of manually reviewing hours of CCTV footage, users can search for specific objects or scenes and instantly retrieve the relevant timestamps where they appear.

Example queries:

- "truck"
- "person wearing a backpack"
- "red car"
- "man carrying a box"

The system aims to provide a faster and more efficient way to analyze surveillance footage using modern Computer Vision and Multimodal AI techniques.

---

## 🎯 Problem Statement

Reviewing CCTV recordings is a time-consuming process, especially when searching for a specific object or event within hours of footage.

VisionSeek aims to solve this problem by enabling semantic search over video content, allowing users to find relevant moments using simple text descriptions.

---

## 🚀 Current Development Status

This project is currently under active development.

### Completed

- Video frame extraction pipeline
- Initial experiments with CLIP embeddings
- Evaluation of image-text retrieval approaches
- Comparison of CLIP and SigLIP models
- Early prototype testing on CCTV footage

### In Progress

- YOLO-based object detection
- SigLIP embedding generation
- Vector database indexing (FAISS)
- Timestamp retrieval system
- Search engine implementation

### Planned Features

- Natural language video search
- Object-level retrieval
- Timestamp-based navigation
- Thumbnail previews
- Streamlit web application
- Video upload and indexing
- Real-time search results

---

## 🏗 Planned Architecture

```text
Video Upload
      ↓
Frame Extraction
      ↓
YOLO Object Detection
      ↓
Object Cropping
      ↓
SigLIP Embeddings
      ↓
FAISS Vector Search
      ↓
Natural Language Query
      ↓
Relevant Timestamps & Results
```

---

## 🛠 Technologies

- Python
- OpenCV
- YOLOv8
- SigLIP
- FAISS
- NumPy
- Pandas
- Streamlit

---

## 📂 Project Structure

```text
VisionSeek
│
├── data
├── notebooks
├── src
├── app
├── models
├── requirements.txt
└── README.md
```

---

## 🔮 Future Goals

- Search surveillance videos using natural language
- Support multiple video formats
- Improve retrieval accuracy with object-level indexing
- Generate searchable video embeddings
- Deploy as a web application

---

## 📌 Project Status

🚧 Work in Progress

This repository currently contains research experiments, prototype implementations, and ongoing development toward the first functional MVP.

---

## 👨‍💻 Author

**Sujith V S**

Building VisionSeek to explore the intersection of Computer Vision, Multimodal AI, and Video Retrieval Systems.