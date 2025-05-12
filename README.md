# Dimensionality Reduction and Classification in Remote Sensing Images

## 📘 Overview

This project focuses on applying dimensionality reduction techniques to hyperspectral remote sensing images, followed by classification using the K-Nearest Neighbors (KNN) algorithm. It uses a novel method called **Kernel Subspace PCA (SubXPCA)** to reduce data dimensionality while retaining important features, resulting in better classification performance and improved model accuracy.

---

## 🚀 Features

- Dimensionality reduction using Kernel Subspace PCA
- Classification using K-Nearest Neighbors (KNN)
- Web interface built with Flask
- Real-time accuracy prediction on uploaded `.csv` datasets
- Modular architecture with frontend and backend components

---

## 🧠 Technologies Used

- **Python**
- **Flask** – Web framework
- **Scikit-learn** – Machine learning
- **Pandas & NumPy** – Data processing
- **HTML/CSS** – Frontend UI

---

## 📊 How It Works

1. Upload a `.csv` file containing spectral data (last column should be class labels).
2. Input:
    - Number of principal components to retain
    - Number of nearest neighbors for classification
3. The app:
    - Applies Kernel PCA on feature partitions
    - Re-merges them and applies SubXPCA
    - Trains and tests a KNN classifier
    - Displays the accuracy of classification


