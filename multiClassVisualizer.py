#!/usr/bin/env python
"""
This script:
  1. Loads an image classification dataset (a folder of folders of images).
  2. Loads a previously trained TinyVGG model.
  3. Computes and displays a confusion matrix.
  4. Displays a grid of images, one per class, with a simple GradCAM-like heatmap overlay
     to highlight important features.
"""

# -------------------------
# Imports
# -------------------------
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
import math

# -------------------------
# Device Setup
# -------------------------
# Use Apple's MPS device if available on Mac Silicon, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# -------------------------
# Model Definition: TinyVGG
# -------------------------
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU()
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU()
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 32 * 32, out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.block_4(self.block_3(self.block_2(self.block_1(x)))))

# -------------------------
# Data Setup
# -------------------------
data_dir = "/Users/rmahshie/Downloads/cs4100/Project/asl_dataset"
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
print("Classes:", class_names)

# Create a DataLoader (for confusion matrix computation)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# -------------------------
# Load Model
# -------------------------
model = TinyVGG(input_shape=3, hidden_units=30, output_shape=len(class_names))
model.load_state_dict(torch.load("modelSaves/tinyvgg_asl_model0.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded and ready.")

# -------------------------
# 1. Compute and Plot Confusion Matrix
# -------------------------
all_preds = []
all_targets = []
with torch.inference_mode():
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

cm = confusion_matrix(all_targets, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# -------------------------
# 2. Visualize Salient Features (One Sample per Class)
# -------------------------
def generate_heatmap(model, input_img, target_layer):
    """
    A simple GradCAM-like function.
    Registers a forward hook on `target_layer` to capture its output,
    runs a forward pass with the input image, and then averages the activations
    over channels to generate a heatmap.
    
    Returns a numpy array (H x W) normalized between 0 and 1.
    """
    model.eval()
    activations = {}

    def hook_fn(module, input, output):
        activations['value'] = output.detach()

    hook = target_layer.register_forward_hook(hook_fn)
    
    with torch.inference_mode():
        _ = model(input_img.unsqueeze(0).to(device))
    
    hook.remove()

    act = activations['value']  # shape: [1, C, H, W]
    heatmap = act.mean(dim=1).squeeze()  # average along channels, shape: [H, W]
    # Normalize the heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.cpu().numpy()

def overlay_heatmap_on_image(image, heatmap, alpha=0.4, colormap=plt.cm.jet):
    """
    Overlays a heatmap on an image.
    image: numpy array of shape (H, W, C) with values in [0,1].
    heatmap: numpy array of shape (H, W) with values in [0,1].
    Returns an overlayed image as a numpy array.
    """
    heatmap_color = colormap(heatmap)[:, :, :3]  # convert heatmap to RGB by dropping alpha channel
    overlay = alpha * heatmap_color + (1 - alpha) * image
    overlay = np.clip(overlay, 0, 1)
    return overlay

# Collect one sample per class from the dataset.
num_classes = len(class_names)
sample_overlays = {}  # dictionary keyed by class index
for image, label in dataset:
    if label not in sample_overlays:
        # Use target_layer: for example, the first conv layer in block_4
        target_layer = model.block_4[0]
        heatmap = generate_heatmap(model, image, target_layer)
        # Resize heatmap to image dimensions using interpolation
        heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)  # shape: [1,1,H,W]
        heatmap_resized = nn.functional.interpolate(heatmap_tensor, size=(image.shape[1], image.shape[2]), mode='bilinear', align_corners=False)
        heatmap_resized = heatmap_resized.squeeze().numpy()
        
        # Convert image from [C, H, W] to [H, W, C]
        image_np = image.permute(1, 2, 0).numpy()
        overlay = overlay_heatmap_on_image(image_np, heatmap_resized, alpha=0.4)
        
        sample_overlays[label] = overlay
        
    if len(sample_overlays) == num_classes:
        break

# Plot each class's overlay in a grid.
cols = min(num_classes, 6)
rows = math.ceil(num_classes / cols)
fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
fig.suptitle("Representative Image per Class with Heatmap Overlay", fontsize=20)

for idx in range(num_classes):
    row = idx // cols
    col = idx % cols
    if rows > 1:
        ax = axes[row, col]
    else:
        ax = axes[col]
    ax.imshow(sample_overlays[idx])
    ax.set_title(f"Class: {class_names[idx]}")
    ax.axis("off")

# Hide any extra subplots
for idx in range(num_classes, rows * cols):
    row = idx // cols
    col = idx % cols
    if rows > 1:
        axes[row, col].axis("off")
    else:
        axes[col].axis("off")

plt.tight_layout()
plt.show()
