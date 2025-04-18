#!/usr/bin/env python
"""
Combined script for loading an image classification dataset, building the TinyVGG model, 
and training/testing the model. All code is combined into one file without changing the internal code.
"""

# -------------------------
# Imports
# -------------------------
import torch
from torch import nn
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from timeit import default_timer as timer
from tqdm.auto import tqdm

# defining device to be gpu if available
if torch.backends.mps.is_available():
  device = torch.device("mps")
  print("Using MPS device")
else:
  device = torch.device("cpu")
  print("Using CPU device")

# -------------------------
# Data Setup
# -------------------------
data_dir = "/Users/rmahshie/Downloads/cs4100/Project/asl_dataset"

# defining the transform for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# get the data from the folder in tensor form
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# calculate the size of train data and test data
train_size = int(.8 * len(dataset))
test_size = len(dataset) - train_size

# split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

BATCH_SIZE = 32

# creating the iterable dataloaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Check the lengths
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# -------------------------
# Model Definition
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
        nn.Dropout(p=.2),
        nn.Linear(in_features=hidden_units*32*32, out_features=output_shape)
    )

  
  def forward(self, x):
    return self.classifier(self.block_4(self.block_3(self.block_2(self.block_1(x)))))

# -------------------------
# Training Functions
# -------------------------
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
  
  # put model in train mode
  model.train()

  # setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through the data loader data batches
  # (X, y) is a batch of data; for example, X has shape [batch_size, C, H, W] and y [batch_size]
  for batch, (X, y) in enumerate(dataloader):
    
    X, y = X.to(device), y.to(device)

    # forward pass
    y_pred = model(X)

    # calculate and accumulate loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    # optimizer zero grad
    optimizer.zero_grad()

    # loss backwards
    loss.backward()

    # step the optimizer
    optimizer.step()

    # calculate and accumulate accuracy metrics across all batches
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item() / len(y_pred)
  
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
  
  # put model in eval mode
  model.eval()

  # setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

  with torch.inference_mode():
    # loop through the batches
    for batch, (X, y) in enumerate(dataloader):

      X, y = X.to(device), y.to(device)
    
      # forward pass
      test_pred_logits = model(X)

      # calculate and accumulate loss
      loss = loss_fn(test_pred_logits, y)
      test_loss += loss.item()

      # calculate and accumulate accuracy
      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
  
  # adjust metrics to get average loss and accuracy per batch
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
  
  # empty results dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}

  # loop through training and testing steps for epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer)
    
    test_loss, test_acc = test_step(model=model,
                                     dataloader=test_dataloader,
                                     loss_fn=loss_fn)
    
    # print what's happening
    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f}"
    )

    # update the results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
  
  return results

# -------------------------
# Running the Training
# -------------------------
from timeit import default_timer as timer

torch.manual_seed(42)


# set number of epochs
NUM_EPOCHS = 10

# create instance of the model
model_0 = TinyVGG(input_shape=3,
                  hidden_units=30,
                  output_shape=len(dataset.classes))
model_0.to(device)

# setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# start the timer
start_time = timer()

# train model_0 

model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

# end the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")


# Save the model state dict (just the weights)
# keep commented out if you don't want to save this time
#import os

# make sure folder exists
#os.makedirs("modelSaves", exist_ok=True)

#MODEL_SAVE_PATH = "modelSaves/tinyvgg_asl_model0_dropout.pth"
#torch.save(model_0.state_dict(), MODEL_SAVE_PATH)
#print(f"Model saved to {MODEL_SAVE_PATH}")
