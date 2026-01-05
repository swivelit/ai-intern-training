import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from joblib import dump

from utils import ensure_dir, list_images, load_and_preprocess_images


def psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((max_val ** 2) / mse)


def make_grid(original, reconstructed, save_path, n=10, title="Original (top) vs Reconstructed (bottom)"):
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


def main(
    data_dir="data/images",
    out_dir="outputs",
    image_size=(64, 64),
    n_components_list=(20, 50, 100, 200),
    sample_grid_n=10,
    random_seed=42,
):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "plots"))
    ensure_dir(os.path.join(out_dir, "models"))

    paths = list_images(data_dir)
    if len(paths) < 5:
        raise ValueError(
            f"Need at least ~5 images in {data_dir}. Found {len(paths)}.\n"
            "Add more images and re-run."
        )

    imgs = load_and_preprocess_images(paths, size=image_size)  # (N,H,W,C)
    N, H, W, C = imgs.shape
    d = H * W * C

    X = imgs.reshape(N, d)

    # simple split
    rng = np.random.default_rng(random_seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(0.8 * N)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    imgs_test = imgs[test_idx]

    results = []
    best = None

    for n_components in n_components_list:
        print(f"\nTraining PCA with n_components={n_components} ...")
        pca = PCA(n_components=n_components, random_state=random_seed)

        pca.fit(X_train)

        X_hat_test = pca.inverse_transform(pca.transform(X_test))

        mse = mean_squared_error(X_test, X_hat_test)
        p = psnr(mse)
        explained = float(np.sum(pca.explained_variance_ratio_))

        print(f"Explained variance ratio (sum): {explained:.4f}")
        print(f"Test MSE: {mse:.6f} | PSNR: {p:.2f} dB")

        recon_imgs = X_hat_test.reshape(-1, H, W, C)
        grid_path = os.path.join(out_dir, "plots", f"custom_pca_{n_components}_grid.png")
        make_grid(imgs_test, recon_imgs, grid_path, n=sample_grid_n,
                  title=f"Custom Images PCA n_components={n_components} (ExplVar={explained:.3f})")

        model_path = os.path.join(out_dir, "models", f"pca_custom_{n_components}.joblib")
        dump(pca, model_path)

        row = {
            "n_components": int(n_components),
            "explained_variance_sum": explained,
            "test_mse": float(mse),
            "test_psnr_db": float(p),
            "grid_path": grid_path,
            "model_path": model_path,
            "image_size": list(image_size),
            "num_images": int(N),
        }
        results.append(row)

        if best is None or row["test_psnr_db"] > best["test_psnr_db"]:
            best = row

    summary_path = os.path.join(out_dir, "metrics_custom.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, indent=2)

    print("\nSaved:", summary_path)
    print("Best:", best)


if __name__ == "__main__":
    main()
