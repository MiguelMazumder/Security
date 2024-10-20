import h5py
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms  # For data augmentation


# Dataset class for handling key images and bittings
class KeyDataset(Dataset):
    def __init__(self, images, labels, feature_extractor):
        self.images = images
        self.labels = labels
        self.feature_extractor = feature_extractor

        # Define data augmentation transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for ViT input
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),  # Convert image to Tensor after augmentations
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Ensure the image is properly scaled
        img = self.images[idx]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)  # Scale to [0, 255] if needed

        img = Image.fromarray(img)  # Convert the NumPy array to a PIL image
        key_integer = self.labels[idx]

        # Apply transformations
        img = self.transform(img)

        # Extract features from the image using ViT's feature extractor
        inputs = self.feature_extractor(images=img, return_tensors="pt")

        # Return pixel values and the corresponding labels (key integers)
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': torch.tensor(key_integer, dtype=torch.float32)
        }

# Load ViT model and feature extractor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=5,
    ignore_mismatched_sizes=True
)

# Open the HDF5 file to extract key images and bittings
with h5py.File('keynet.h5', 'r') as h5_file:
    obverse = h5_file['obverse'][:]  # Shape: (319, 512, 512, 3)
    bittings = h5_file['bittings'][:]  # Shape: (319, 5)

# Create dataset using obverse images and key_integers as labels
dataset = KeyDataset(obverse, bittings, feature_extractor)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # Increase number of epochs
    per_device_train_batch_size=8,
    eval_strategy="epoch",  # Use 'eval_strategy' instead of 'evaluation_strategy'
    save_strategy="epoch",  # Make sure save strategy matches eval strategy
    save_steps=10,  # Save every 10 steps
    save_total_limit=1,  # Keep only the latest checkpoint
    logging_dir='./logs',
    learning_rate=5e-5,  # Adjust learning rate if needed
    weight_decay=0.01,  # Regularization
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# Trainer for training the ViT model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training the model
trainer.train()
