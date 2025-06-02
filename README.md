# 🧠 Semantic Segmentation Using SegNet Architecture (Tensor)

This project demonstrates a simplified implementation of **semantic segmentation** using a SegNet-style **encoder–decoder convolutional neural network** in **PyTorch**, trained on a **single traffic scene image**. It showcases how pixel-wise classification can be performed using low-level to high-level feature extraction, pooling indices for precise spatial unpooling, and end-to-end learning.

---

## 🔧 What It Does

- Implements a **SegNet-style** architecture (Conv → BatchNorm → ReLU → MaxPool + Indices → Decoder with Unpooling)
- Trains from scratch on a **single uploaded image** using **Google Colab**
- Generates a **synthetic binary segmentation mask** by thresholding the grayscale version of the image
- Visualizes both the **original image** and the **predicted segmentation mask**

---

## 🚀 Features

- Fully runnable in **Google Colab**
- Manual image upload support via `files.upload()`
- Automatic resizing to **256×256**
- Lightweight training loop using **CrossEntropyLoss**
- Segmentation mask output visualization with **matplotlib**

---

## 📁 Files

- `segnet_colab.ipynb` – Colab notebook with model code, training, and visualization
- `README.md` – Complete project documentation and usage instructions

---

## 📚 Based On

> **SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation**  
> Badrinarayanan, Kendall, Cipolla (2017)  
> [arXiv:1511.00561](https://arxiv.org/abs/1511.00561)

---

## 🛠️ Tech Stack

- Python 3
- PyTorch
- Pillow
- NumPy
- Matplotlib
- Google Colab

---

## 📄 License

This project is open-source under the **MIT License**.
