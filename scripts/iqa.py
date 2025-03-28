# -*- coding: utf-8 -*-

"""
Generate and save image quality assessment (IQA) scores for a single image using various metrics.
"""

import warnings

# Suppress most future deprecation warnings, user warnings, etc.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pyiqa
import torch
import json
from PIL import Image
from torchvision import transforms


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
IMAGE_PATH = "5ffba245-0ea2-4d09-8a97-b7729ad5c988.png"        # <-- Replace with your test image path
OUTPUT_JSON = "iqa_nr_results.json"  # Where to store final results


# -------------------------------------------------------------
# Setup
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess image
transform = transforms.Compose([transforms.ToTensor()])
image = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(image).unsqueeze(0).to(device)

# For logging final results
results = {}

# Known error substrings we want to skip or handle gracefully
SKIP_ERRORS = [
    "Please specify reference image",  # FR metrics
    "When fdir2 is not provided",      # FID references
    "No face detected in the input image",
    "caption_list is None",           # Clipscore-like
    "only integer tensors of a single element",
    "`.to` is not supported for `4-bit` or `8-bit`",  # QAlign 4-bit
]

# -------------------------------------------------------------
# Metric Loop
# -------------------------------------------------------------
all_metrics = pyiqa.list_models()

for name in all_metrics:
    try:
        metric = pyiqa.create_metric(name, device=device)

        # Attempt to compute with single image (NR style).
        try:
            score = metric(img_tensor)
            results[name] = float(score.item())
            print(f"{name}: {score.item():.4f}")

        except Exception as e:
            error_msg = str(e)
            # Check if it matches any known skip pattern
            if any(substring in error_msg for substring in SKIP_ERRORS):
                print(f"Skipping {name} -> {error_msg}")
                results[name] = f"skipped: {error_msg}"
            else:
                # Otherwise record the unknown error
                print(f"Error computing {name} -> {error_msg}")
                results[name] = f"error: {error_msg}"

    except Exception as e:
        # If the metric can't even be created, log that
        print(f"Failed to load/create {name} -> {str(e)}")
        results[name] = f"error: {str(e)}"


# -------------------------------------------------------------
# Save results to JSON
# -------------------------------------------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\u2705 Saved results to: {OUTPUT_JSON}")
