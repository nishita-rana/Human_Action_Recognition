Human Activity Recognition using CNN and LSTM
📌 Overview

This project implements Human Activity Recognition (HAR) using a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture.
The model is trained on the UCF50 dataset, which contains 50 human action categories. The goal is to classify human activities from video sequences by leveraging both spatial features (via CNNs) and temporal dynamics (via LSTMs/ConvLSTMs).

Applications include:

🏥 Healthcare – patient monitoring, fall detection

👮 Public safety – surveillance & anomaly detection

⚽ Sports analytics – performance monitoring

🖥 Human-computer interaction – gesture/activity recognition

📂 Dataset

Dataset Used: UCF50 Human Action Dataset

Activities Considered (subset):

Downstairs

Jogging

Sitting

Standing

Upstairs

Walking

⚙ Data Preprocessing

The dataset undergoes several preprocessing steps before training:

Label Encoding – Converts activity names into numeric labels.

Linear Interpolation – Handles missing values in frames.

Data Split – 70% training, 15% validation, 15% testing.

Normalization – Scales pixel values (0–1).

Segmentation – Splits video into equal-sized clips.

One-Hot Encoding – Converts labels into binary vectors for multi-class classification.

🏗 Model Architecture

The model is a ConvLSTM-based hybrid designed for spatiotemporal learning:

Input Layer: Video frames (T, H, W, 3)

ConvLSTM2D Layers (64, 128, 256 filters) – Extract temporal and spatial features

MaxPooling3D Layers – Reduce spatial dimensions

Dropout Layers – Prevent overfitting

Flatten Layer – Converts 5D tensor to 1D feature vector

Dense + Softmax Output – Classifies into activities

🚀 Installation & Usage
🔧 Requirements

Python 3.8+

TensorFlow / Keras

NumPy, Pandas, Matplotlib

OpenCV
