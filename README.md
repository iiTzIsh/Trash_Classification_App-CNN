# Trash Classification App - CNN

## 🌟 Introduction
The **Trash Classification App** is a machine learning-powered tool that classifies images of trash into six categories. It provides proper disposal instructions and eco-friendly tips to promote sustainable waste management.

---

## 🔗 Links
- **GitHub Repository:** [Trash_Classification_App-CNN](https://github.com/iiTzIsh/Trash_Classification_App-CNN)  
- **Dataset Source:** [Kaggle - TrashNet Dataset](https://www.kaggle.com/datasets/feyzazkefe/trashnet)  
- **Live Demo:** [Streamlit App](https://ish-trash-classify.streamlit.app/)

---

## 📷 Features
1. **Upload Trash Images:** Upload an image of trash (e.g., plastic, cardboard).  
2. **Predict Trash Type:** The app classifies the image with high accuracy.  
3. **Disposal Tips:** Provides instructions on how to dispose of or recycle the item.  
4. **Eco Tips:** Offers actionable advice to reduce waste.

---

## 🧠 Model Overview
The app uses a **MobileNetV2 CNN model** fine-tuned with **transfer learning** for high performance. Key techniques include:
1. **Transfer Learning:** Pretrained on ImageNet and fine-tuned for trash classification.
2. **Data Augmentation:** Techniques like rotation, flipping, and zooming to improve robustness.
3. **Batch Normalization and Dropout:** Enhances model stability and prevents overfitting.
4. **Learning Rate Scheduling:** Dynamically adjusts learning rate for better convergence.

**Specifications:**
- **Input Shape:** `(224, 224, 3)`  
- **Output Classes:** `Cardboard`, `Glass`, `Metal`, `Paper`, `Plastic`, `Trash`  
- **Frameworks Used:** TensorFlow/Keras and Streamlit  

**Performance:** Achieved ~95% validation accuracy with early stopping.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iiTzIsh/Trash_Classification_App-CNN.git
   cd Trash_Classification_App-CNN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---


## 🌱 Eco Tip
"Carry reusable bags, bottles, and containers to reduce waste. Small actions make a big impact! 🌍"

---
