import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from joblib import dump


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((max_val ** 2) / mse)


def make_grid(original, reconstructed, save_path, n=10, title="Original (top) vs Reconstructed (bottom)"):
    # original/reconstructed: (N, H, W, C), values in [0,1]
    n = min(n, original.shape[0])
    fig = plt.figure(figsize=(n * 1.2, 2.8))
    plt.suptitle(title)

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        ax.imshow(original[i])
        ax.axis("off")

        ax = plt.subplot(2, n, n + i + 1)
        ax.imshow(np.clip(reconstructed[i], 0, 1))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def load_cifar10():
    """
    Tries to load CIFAR-10 via TensorFlow Keras if available.
    If not available, raises a clear error with instructions.
    """
    try:
        from tensorflow.keras.datasets import cifar10
        (x_train, _), (x_test, _) = cifar10.load_data()
        return x_train, x_test
    except Exception as e:
        raise RuntimeError(
            "Could not load CIFAR-10. Install tensorflow OR use the custom images script.\n"
            "Install: pip install tensorflow\n"
            f"Original error: {e}"
        )


def main(
    out_dir="outputs",
    n_components_list=(50, 100, 200, 400),
    sample_grid_n=12,
    max_train=20000,
    max_test=2000,
    random_seed=42,
):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "plots"))
    ensure_dir(os.path.join(out_dir, "models"))

    np.random.seed(random_seed)

    x_train, x_test = load_cifar10()

    # Normalize to [0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Subsample for speed
    x_train = x_train[:max_train]
    x_test = x_test[:max_test]

    H, W, C = x_train.shape[1:]
    d = H * W * C

    X_train = x_train.reshape(len(x_train), d)
    X_test = x_test.reshape(len(x_test), d)

    results = []
    best = None

    for n_components in n_components_list:
        print(f"\nTraining PCA with n_components={n_components} ...")
        pca = PCA(n_components=n_components, random_state=random_seed)

        Z_train = pca.fit_transform(X_train)
        X_hat_test = pca.inverse_transform(pca.transform(X_test))

        mse = mean_squared_error(X_test, X_hat_test)
        p = psnr(mse)

        explained = float(np.sum(pca.explained_variance_ratio_))

        print(f"Explained variance ratio (sum): {explained:.4f}")
        print(f"Test MSE: {mse:.6f} | PSNR: {p:.2f} dB")

        # Save grid
        recon_imgs = X_hat_test.reshape(-1, H, W, C)
        grid_path = os.path.join(out_dir, "plots", f"cifar10_pca_{n_components}_grid.png")
        make_grid(x_test, recon_imgs, grid_path, n=sample_grid_n,
                  title=f"CIFAR-10 PCA n_components={n_components} (ExplVar={explained:.3f})")

        # Save model
        model_path = os.path.join(out_dir, "models", f"pca_cifar10_{n_components}.joblib")
        dump(pca, model_path)

        row = {
            "n_components": int(n_components),
            "explained_variance_sum": explained,
            "test_mse": float(mse),
            "test_psnr_db": float(p),
            "grid_path": grid_path,
            "model_path": model_path,
        }
        results.append(row)

        if best is None or row["test_psnr_db"] > best["test_psnr_db"]:
            best = row

    # Save summary JSON
    summary_path = os.path.join(out_dir, "metrics_cifar10.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, indent=2)

    print("\nSaved:", summary_path)
    print("Best:", best)


if __name__ == "__main__":
    main()
