import cv2
import os

def load_images(folder):
    images = []
    filenames = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)

        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
            filenames.append(file)

    return images, filenames


def save_image(path, image):
    cv2.imwrite(path, image)