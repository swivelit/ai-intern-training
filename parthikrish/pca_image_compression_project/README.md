
# Image Compression using PCA

## Dataset
Uses free sample images bundled with scikit-learn (`china.jpg`, `flower.jpg`).

## Steps
1. Load images
2. Apply PCA channel-wise
3. Reduce dimensions
4. Reconstruct images
5. Compare original vs compressed

## Run
```bash
pip install numpy matplotlib scikit-learn
python scripts/pca_image_compression.py
```

## Output
Comparison images are saved in `outputs/`
