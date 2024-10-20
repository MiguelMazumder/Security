import h5py
import torch
import os
import glob
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np

# Function to predict key integers using the trained model
def predict_bitting(images, model, feature_extractor):
    model.eval()  # Set model to evaluation mode
    predictions = []

    with torch.no_grad():
        for img in images:
            img_pil = Image.fromarray(img.astype('uint8'))  # Keep RGB format
            inputs = feature_extractor(images=img_pil, return_tensors="pt")
            outputs = model(**inputs)
            pred_key_integer = outputs.logits.squeeze().cpu().numpy()
            predictions.append(pred_key_integer)

    return np.array(predictions)

# Load the reverse images and bittings from the HDF5 file
with h5py.File('keynet.h5', 'r') as h5_file:
    reverse = h5_file['reverse'][:20]  # Shape: (319, 512, 512, 3)
    bittings = h5_file['bittings'][:20]  # Shape: (319, 5)

# Function to get the latest checkpoint directory
def get_latest_checkpoint(dir_path):
    # Use glob to find all checkpoint directories
    checkpoint_paths = glob.glob(os.path.join(dir_path, 'checkpoint-*'))
    if not checkpoint_paths:
        raise ValueError("No checkpoints found in the specified directory.")
    
    # Sort the checkpoints and return the latest one
    latest_checkpoint = max(checkpoint_paths, key=os.path.getctime)  # Get the most recently created checkpoint
    return latest_checkpoint

# Load the latest ViT model from the checkpoint
latest_checkpoint = get_latest_checkpoint('./results/')
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(latest_checkpoint, num_labels=5)
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Predict key integers using the reverse images
predictions = predict_bitting(reverse, model, feature_extractor)

matches = []

# Iterate through the predicted key integers
for idx, pred in enumerate(predictions):
    # Check against all original key integers
    for current_bitting in bittings:
        # Compare with the original key integer
        if np.all(np.isclose(pred, current_bitting, atol=1)):  # Allow small differences
            matches.append((idx, pred, current_bitting))

# Output the matches found
if matches:
    print("Found matches between predicted key integers from reverse images and original key integers:")
    for match in matches:
        print(f"Reverse image index: {match[0]}, Predicted: {match[1]}, Original key integer: {match[2]}")
else:
    print("No matches found between predicted key integers from reverse images and original key integers.")

# Print the first few predicted key integers
print("Predictions (first 5):")
print(predictions[:5])

# Print the first few original key integers
print("Original key integers (first 5):")
print(bittings[:5])
