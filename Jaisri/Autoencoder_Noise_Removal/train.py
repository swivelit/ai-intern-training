import os
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data
from model import build_autoencoder

def main():
    # 1. Load data
    print("Loading and adding noise to MNIST...")
    x_train, x_train_noisy, x_test, x_test_noisy = load_and_preprocess_data()

    # 2. Build and Train Model
    # Since we are on CPU, we'll limit the training significantly to prevent lag
    print("Building model...")
    autoencoder = build_autoencoder()
    
    print("Starting training (optimizing on CPU with small subset for demo)...")
    # Using only 10,000 samples and 2 epochs for quick demonstration on CPU
    autoencoder.fit(x_train_noisy[:10000], x_train[:10000],
                    epochs=2,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))

    # 3. Evaluate and Predict
    print("Denoising test images...")
    denoised_images = autoencoder.predict(x_test_noisy[:10])

    # 4. Save and Plot Comparison
    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("Original")

        # Noisy
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("Noisy Input")

        # Denoised
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(denoised_images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("Autoencoder Output")

    save_path = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Autoencoder_Noise_Removal/denoising_results.png"
    plt.savefig(save_path)
    print(f"Results plot saved to {save_path}")

    # 5. Save model
    model_save_path = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Autoencoder_Noise_Removal/noise_remover_model.h5"
    autoencoder.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
