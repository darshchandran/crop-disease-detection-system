# Crop Disease Detection System

Agriculture loses billions every year to crop diseases that go undetected until it's too late. This project was my attempt to see if a computer vision model could spot disease in plant leaves from just a photo — the kind of thing a farmer could eventually use from their phone.

Transfer learning made this possible without needing months of training time or a GPU farm.

---

## What it does

Takes a photo of a plant leaf as input and classifies it into one of 38 categories — covering both healthy plants and various disease types across multiple crops.

```
Input:  [Photo of tomato leaf]
Output: Tomato — Late Blight (Confidence: 94.3%)

Input:  [Photo of corn leaf]
Output: Corn — Healthy (Confidence: 97.1%)
```

---

## Why I built it

I had already built a basic agricultural information website and wanted to take it further — what if the site could actually diagnose crop problems instead of just displaying information? This project was the ML backbone for that idea. It also taught me transfer learning, which is one of the most practically useful techniques in deep learning.

---

## Dataset

**PlantVillage Dataset** — 87,000 images of healthy and diseased plant leaves across 38 classes.  
Covers 14 crop species including tomato, potato, corn, grape, apple and more.  
Available on Kaggle: [Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## Tech Stack

- **Python 3.11**
- **PyTorch / Torchvision** — model building, transfer learning, training loop
- **MobileNetV2** — pretrained CNN used as the base model
- **Pillow** — image loading and preprocessing
- **Matplotlib** — visualizing samples, training curves, confusion matrix
- **NumPy** — array operations
- **Jupyter Notebook** — development environment

---

## How it works

### 1. Load & Explore
Dataset comes pre-organized into folders by class. Used PyTorch's `ImageFolder` to load it. Plotted sample images from each class to understand what diseased vs healthy leaves look like visually.

### 2. Preprocessing & Augmentation
- Resized all images to 224x224 (MobileNetV2's expected input size)
- Normalized using ImageNet mean and std values
- Applied data augmentation on training set — random horizontal flips, rotations, color jitter — to make the model more robust to real-world photo conditions

### 3. Transfer Learning with MobileNetV2
Instead of training a CNN from scratch (which would take days), loaded MobileNetV2 pretrained on ImageNet — it already knows how to detect edges, textures, and shapes from 1.2 million images. Froze all the base layers and only replaced the final classification head for our 38 classes.

```python
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, 38)
```

### 4. Train
- **Loss:** CrossEntropy
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 15
- Trained only the new classification head — fast and efficient

### 5. Evaluate
- Plotted training vs validation accuracy and loss curves
- Generated confusion matrix across all 38 classes
- Checked per-class accuracy to spot which diseases were hardest to detect

### 6. Custom prediction
Drop in any leaf photo — outputs disease name and confidence score.

---

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 94.2% |
| Test Accuracy | 93.8% |
| Training Time | ~25 mins (CPU) |

The model does well on most classes but occasionally confuses diseases that look visually similar — early blight vs late blight on tomatoes, for example. Makes sense — even experienced farmers struggle with those.

---

## How to run it

```bash
git clone https://github.com/yourusername/crop-disease-detector.git
cd crop-disease-detector

pip install torch torchvision Pillow matplotlib numpy jupyter

jupyter notebook crop_disease_detection.ipynb
```

Download the PlantVillage dataset from Kaggle and place it in a `data/` folder in the root directory before running.

---

## Project structure

```
crop-disease-detector/
│
├── crop_disease_detection.ipynb    # Main notebook — run this
├── data/                           # PlantVillage dataset (download from Kaggle)
│   ├── train/
│   ├── valid/
│   └── test/
├── model/
│   └── mobilenet_plant_disease.pth # Saved trained model
├── test_images/                    # Custom leaf images for testing
├── requirements.txt
└── README.md
```

---

## What I learned

- How transfer learning works and why it's practical for real-world CV problems
- MobileNetV2 architecture and why it's efficient for mobile/edge deployment
- Data augmentation and why it matters for image datasets
- How to freeze layers and fine-tune only what you need
- Reading a multi-class confusion matrix (38 classes is a lot to interpret)

---

## What's next

- Fine-tune the base layers too (full fine-tuning) for higher accuracy
- Build a Streamlit or Flask web app — upload a leaf photo, get a diagnosis
- Compress the model using quantization for mobile deployment
- Connect back to the agricultural website project as an actual diagnostic feature
