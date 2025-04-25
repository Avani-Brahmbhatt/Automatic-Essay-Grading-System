# Automatic-Essay-Grading-System

---

# 📚 Automated Essay Grading System using RoBERTa

An AI-powered essay grading system designed to automatically evaluate student essays and assign scores (1–6) based on semantic quality, structure, coherence, and language. This project uses the **RoBERTa** language model fine-tuned on a dataset of over 25,000 essays, along with custom data augmentation strategies to enhance performance on underrepresented score classes.

## 🚀 Key Features

- **RoBERTa-based Fine-Tuning**: Utilizes a pretrained RoBERTa model for context-aware essay scoring.
- **Gradio Interface**: Provides a user-friendly frontend where users can input essays and receive a predicted score instantly.
- **Balanced Dataset**: Includes synthetic data generation via back-translation and synonym replacement to mitigate class imbalance.
- **Explainable Predictions** *(optional)*: Highlights important parts of the essay influencing the model’s decision.

---

## 🧠 Model Overview

- **Base Model**: `roberta-base`
- **Input**: Raw student essays (text)
- **Output**: A score between 1 and 6
- **Dataset**: Kaggle open-source essay scoring dataset (17,000+ entries) + augmented data
- **Loss Handling**: Weighted loss function to handle class imbalance

---

## 🧪 Data Augmentation Techniques

Implemented in `roberta-essay-augmentation.ipynb`:
- **Back-Translation**: Essays are translated to another language and back to generate paraphrased versions.
- **Synonym Replacement**: Randomly replaces words with context-aware synonyms using NLP libraries (e.g., WordNet, contextual embeddings).
- **Score Balancing**: Ensures rare score classes have enough samples through augmentation.

---
## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Avani-Brahmbhatt/Automatic-Essay-Grading-System.git
cd Automatic-Essay-Grading-System
```
### 2. Install Requirements

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
python app.py
```

---

## 📊 Evaluation Metrics

- **Accuracy**
- **F1-Score (Macro & Weighted)**
- **Confusion Matrix** to visualize score-wise performance

---
