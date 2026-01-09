import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
IMG_SIZE = (64, 64)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- LOAD IMAGES ----------
if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

images = []
filenames = []

files = [f for f in os.listdir(IMAGE_DIR)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if len(files) < 2:
    raise ValueError("At least 2 images are required for PCA")

for file in files:
    img = Image.open(os.path.join(IMAGE_DIR, file)).convert("L")
    img = img.resize(IMG_SIZE)
    images.append(np.array(img).flatten())
    filenames.append(file)

# ---------- CREATE DATA MATRIX ----------
X = np.array(images)

print("Total images:", X.shape[0])
print("Pixels per image:", X.shape[1])

# ---------- PCA (IMPORTANT FIX HERE) ----------
N_COMPONENTS = min(50, X.shape[0])

pca = PCA(n_components=N_COMPONENTS)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)

# ---------- SAVE RECONSTRUCTED IMAGES ----------
for i, img in enumerate(X_reconstructed):
    reconstructed = img.reshape(IMG_SIZE)
    Image.fromarray(reconstructed.astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, filenames[i])
    )

print("PCA Compression Completed")
print("Original shape:", X.shape)
print("Compressed shape:", X_reduced.shape)
