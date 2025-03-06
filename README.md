# Multi-Task Sentence Transformer

## 📌 Project Overview

This repository implements a **Multi-Task Sentence Transformer Model** using PyTorch and Hugging Face's Transformers library. The model is designed to:
- Encode sentences into **fixed-length embeddings**.
- Perform **sentence classification** (Task A).
- Conduct **sentiment analysis** (Task B).

The implementation includes **training, inference, and deployment** using **Docker and GitHub Actions**.

---

## ⚙️ Setup and Installation

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/palakrajdevneu/takehome_assessment_fetch.git
cd takehome_assessment_fetch
```

### 2️⃣ **Create a Virtual Environment (Optional)**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## 🚀 Running the Model

### **Train the Model**
```sh
python src/train.py
```

### **Run Inference**
```sh
python src/inference.py
```

---

## 🐳 Docker Usage

### 1️⃣ **Build the Docker Image**
```sh
docker build -t multitask-transformer .
```

### 2️⃣ **Run the Docker Container**
```sh
docker run --rm -it multitask-transformer
```

### 3️⃣ **Pull from Docker Hub**
If the Docker image was built via GitHub Actions, pull and run it using:
```sh
docker pull YOUR_DOCKER_USERNAME/ml-takehome-assignment:latest
docker run --rm -it YOUR_DOCKER_USERNAME/ml-takehome-assignment:latest
```

---

## 📂 Running in Jupyter Notebook

### 1️⃣ **Install Jupyter Notebook (if not installed)**
```sh
pip install notebook
```

### 2️⃣ **Run the Notebook**
```sh
jupyter notebook
```
Open `notebook/FetchML_Apprentice_TakeHome.ipynb` and run all cells for testing.

---

## 🔄 Transfer Learning Approach

### **Choosing a Pre-Trained Model**
- DistilBERT, BERT, or RoBERTa for strong language representations.
- Example: DistilBERT for a balance of speed & accuracy.

### **Freezing and Unfreezing Layers**
- Start by freezing transformer layers to prevent overfitting.
- Gradually unfreeze layers to adapt the model to domain-specific data.

### **Why This Works**
- Prevents loss of general knowledge.
- Efficient training with controlled adaptation to new data.

---

## 📚 Multi-Task Learning Training Loop

### **Handling Data**
- Uses a synthetic dataset for testing.
- Each batch consists of:
  - Sentence text
  - Classification label
  - Sentiment label

### **Forward Pass and Loss Computation**
- Uses Masked Mean Pooling for sentence embeddings.
- Computes classification & sentiment loss separately.
- Combines both losses into a single objective.

### **Tracking Metrics**
- Tracks accuracy for classification & sentiment tasks separately.

---

## ⚡ Training Considerations

### **Scenario 1: Freezing the Entire Network**
✅ Prevents all parameters from updating during training.
✅ Acts as a fixed feature extractor.
✔️ **Pros:** Fast training, low overfitting risk.
✔️ **When to Use:** If working with a very small dataset or a similar domain as the pre-trained model.

### **Scenario 2: Freezing Only the Transformer Backbone**
✅ Keeps transformer layers fixed, fine-tunes only task-specific heads.
✔️ **Pros:** Preserves general language features, reduces overfitting.
✔️ **When to Use:** When pre-trained model features are already useful, but task-specific adaptation is needed.

### **Scenario 3: Freezing One Task-Specific Head**
✅ Only updates either the classification or sentiment head.
✔️ **Pros:** Protects one task while improving the other.
✔️ **When to Use:** If one task performs well while the other needs fine-tuning.

---

## 🤝 Contributing

We welcome contributions! If you would like to contribute:

1️⃣ **Fork the repository.**
```sh
git clone https://github.com/YOUR_USERNAME/takehome_assessment_fetch.git
```

2️⃣ **Create a new branch:**
```sh
git checkout -b feature-branch
```

3️⃣ **Make your changes and commit them:**
```sh
git commit -m "Add new feature"
```

4️⃣ **Push to the branch:**
```sh
git push origin feature-branch
```

5️⃣ **Open a Pull Request.**

---

## 📜 License

This project is licensed under the **MIT License**.

---
