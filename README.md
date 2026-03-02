Face Recognition Pipeline (DeepFace + OpenCV):

End-to-end facial verification pipeline using deep metric learning and computer vision.

This project detects faces, generates 128D embeddings using FaceNet, and verifies identity using cosine similarity.




What It Does

Recursively scans image datasets
Detects faces using OpenCV (Haar Cascade)
Generates embeddings via DeepFace with FaceNet
Computes cosine distance for identity verification
Supports reference-based and full pairwise comparisons




How It Works:
Detect face
Convert face → 128D embedding
Compute cosine similarity
Classify as MATCH / NO MATCH
Lower distance → same identity
Higher distance → different identities



Example Output
[MATCH] img1.jpg <-> img2.jpg | Dist: 0.1788
[NO MATCH] img1.jpg vs img3.jpg | Dist: 0.9100




Engineering Focus
Handles nested datasets using recursive scanning
Graceful failure handling (enforce_detection=False)



Demonstrates understanding of:
CNN embeddings
Deep metric learning
O(n²) scaling in pairwise comparisons




Tech Stack
Python
OpenCV
DeepFace
NumPy
TesorFlow





Why This Project
This project demonstrates:
Applied computer vision
Embedding-based similarity search

Real-world ML pipeline design

System scalability awareness
