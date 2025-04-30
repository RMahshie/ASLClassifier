# ASL Image Classifier 🧠🤟

A deep learning project that classifies American Sign Language (ASL) hand signs using a custom convolutional neural network (CNN) built in PyTorch. This model was trained on a labeled image dataset and achieves high accuracy across 36 ASL classes.

## 📌 Features

- Custom TinyVGG CNN architecture built from scratch in PyTorch  
- Trained on a dataset of over 5,000 labeled ASL images  
- Supports classification of letters A-Z and digits 0-9  
- Achieves ~98% test accuracy after 10 epochs  
- Includes data preprocessing with PyTorch `transforms` and `ImageFolder`  
- Simple training pipeline with modular code structure

## 🛠️ Tech Stack

- Python  
- PyTorch  
- NumPy  
- Matplotlib (optional, for visualization)  
- torchvision  
- Jupyter Notebook / VS Code  

## 📂 Project Structure

```bash
├── data/                   # ASL image dataset (not included)
├── model/                  # Trained model weights (optional)
├── train.py                # Model training script
├── model.py                # CNN architecture (TinyVGG)
├── utils.py                # Helper functions for training
└── README.md               # You're here!
