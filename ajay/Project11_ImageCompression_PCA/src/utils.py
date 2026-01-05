import os
import cv2
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_images(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for name in os.listdir(folder):
        if name.lower().endswith(exts):
            files.append(os.path.join(folder, name))
    return sorted(files)


def load_and_preprocess_images(image_paths, size=(64, 64)):
    imgs = []
    for p in image_paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        im = im.astype(np.float32) / 255.0
        imgs.append(im)
    if len(imgs) == 0:
        raise ValueError("No valid images found to process.")
    return np.stack(imgs, axis=0)  # (N,H,W,C)
