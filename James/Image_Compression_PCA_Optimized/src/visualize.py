import os
import matplotlib.pyplot as plt
from PIL import Image

original_dir = "data/images"
compressed_dir = "output"

files = os.listdir(original_dir)[:3]

for f in files:
    orig = Image.open(os.path.join(original_dir, f)).convert("L")
    comp = Image.open(os.path.join(compressed_dir, f))

    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(orig, cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Compressed")
    plt.imshow(comp, cmap="gray")
    plt.axis("off")

    plt.show()