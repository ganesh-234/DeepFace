Face Recognition Pipeline (DeepFace + OpenCV)

An end-to-end facial verification pipeline built using deep metric learning and computer vision.
This project detects faces, generates 128-dimensional embeddings using FaceNet, and verifies identity using cosine similarity.
________________________________________

Overview

This system:
•	Recursively scans structured image datasets
•	Detects faces using OpenCV (Haar Cascade)
•	Generates embeddings using DeepFace with FaceNet
•	Computes cosine distance for identity verification
•	Supports both reference-based and full pairwise comparison modes
________________________________________

How It Works
1.	Detect face in image
2.	Convert detected face → 128D embedding
3.	Compute cosine similarity between embeddings
4.	Classify result as MATCH or NO MATCH
•	Lower cosine distance → Same identity
•	Higher cosine distance → Different identities
________________________________________

Example Output
[MATCH] img1.jpg <-> img2.jpg | Dist: 0.1788
[NO MATCH] img1.jpg vs img3.jpg | Dist: 0.9100
________________________________________

Engineering Highlights
•	Recursive dataset handling (supports nested folders)
•	Graceful failure handling (enforce_detection=False)
•	Demonstrates understanding of:
o	CNN-based embeddings
o	Deep metric learning
o	Vector similarity search
o	O(n²) scaling in pairwise comparisons
________________________________________

Tech Stack
•	Python
•	OpenCV
•	DeepFace
•	NumPy
TesorFlow
________________________________________


Why This Project
This project demonstrates applied computer vision, embedding-based similarity systems, and real-world ML pipeline design with scalability awareness.
