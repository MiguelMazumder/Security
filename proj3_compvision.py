import h5py
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

# There are key bittings and images and bittings
class KeyDataset(Dataset):
    def __init__(self, images, labels, feature_extractor):
        self.images = images
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and corresponding bitting label
        img = Image.fromarray(self.images[idx].astype('uint8'))
        bitting = self.labels[idx]

        # Preprocess the image
        inputs = self.feature_extractor(images=img, return_tensors="pt")

        # Return inputs and bitting as label (converted to tensor)
        return {'pixel_values': inputs['pixel_values'].squeeze(), 'labels': torch.tensor(bitting, dtype=torch.float32)}

# Load the ViT model and feature extractor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=5, ignore_mismatched_sizes=True)

# Open the HDF5 file and extract the obverse images and bittings (labels)
with h5py.File('keynet.h5', 'r') as h5_file:
    obverse = h5_file['obverse'][:]  # Shape: (319, 512, 512, 3)
    bittings = h5_file['bittings'][:]  # Shape: (319, 5)

# Create dataset
dataset = KeyDataset(obverse, bittings, feature_extractor)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10,
    save_total_limit=2,
    logging_dir='./logs',
)

# Use Hugging Face's Trainer to train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start training
trainer.train()
