# 🏃 Human Activity Recognition using CNN + LSTM

This project implements **Human Activity Recognition (HAR)** using a hybrid **Convolutional Neural Network (CNN)** and **Long Short-Term Memory (LSTM/ConvLSTM)** architecture.  
The model is trained on the **UCF50 dataset**, which contains 50 human action categories.  
The goal is to classify human activities from video sequences by leveraging both **spatial features (via CNNs)** and **temporal dynamics (via LSTMs/ConvLSTMs)**.

---

## 📌 Applications
- 🏥 **Healthcare** – patient monitoring, fall detection  
- 👮 **Public Safety** – surveillance & anomaly detection  
- ⚽ **Sports Analytics** – performance tracking and improvement  
- 🖥 **Human-Computer Interaction** – gesture/activity recognition  

---

## 📂 Dataset
**Dataset Used:** https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50?resource=download-directory  

**Subset of Activities considered in this project:**
- Downstairs  
- Jogging  
- Sitting  
- Standing  
- Upstairs  
- Walking  

---

## ⚙ Data Preprocessing
Before training, the dataset undergoes the following preprocessing steps:
1. **Label Encoding** – Convert activity names into numeric labels  
2. **Linear Interpolation** – Handle missing values in frames  
3. **Data Splitting** – 70% training, 15% validation, 15% testing  
4. **Normalization** – Scale pixel values to `[0,1]`  
5. **Segmentation** – Split video into equal-sized clips  
6. **One-Hot Encoding** – Convert labels into binary vectors  

---

## 🏗 Model Architecture
The HAR model uses a **ConvLSTM-based hybrid design** for spatiotemporal learning.

- **Input Layer:** Video frames `(T, H, W, 3)`  
- **ConvLSTM2D Layers:** 64, 128, 256 filters for spatiotemporal feature extraction  
- **MaxPooling3D Layers:** Reduce spatial dimensions  
- **Dropout Layers:** Prevent overfitting  
- **Flatten Layer:** Convert 5D tensor to 1D vector  
- **Dense + Softmax Output:** Classify into activity categories  

---

## 🚀 Installation & Usage

### 🔧 Requirements
Make sure you have the following installed:
- Python `3.8+`  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib  
- OpenCV  

### 📥 Clone Repository
```bash
git clone https://github.com/nishita-rana/Human_Action_Recognition.git
```
---

## 📊 Results

Achieved high accuracy on UCF50 dataset (values can be updated after final testing).

##🔮 Future Work

Incorporating attention mechanisms for improved accuracy.

Extending to real-time recognition using webcam feed.

Exploring transformer-based models for HAR.
