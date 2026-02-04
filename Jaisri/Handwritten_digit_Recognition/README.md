# Handwritten Digit Recognition using CNN
This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. It uses PyTorch and achieves >98% accuracy.
## Project Structure
- `main.py`: Main script containing model definition, training, evaluation, and visualization.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.
- `mnist_cnn_model.pth`: Saved model weights (generated after training).
- `*.png`: Visualization results (generated after training).
## Requirements
- Python 3.13+
- PyTorch
- Torchvision
- Matplotlib
- NumPy
Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage
Run the main script:
```bash
python main.py
```
## Model Architecture
The model is a plain CNN with the following structure:
1. **Conv Block 1**: Conv2D (32 filters) -> BatchNorm -> ReLU -> Conv2D (32 filters) -> BatchNorm -> ReLU -> MaxPool -> Dropout
2. **Conv Block 2**: Conv2D (64 filters) -> BatchNorm -> ReLU -> Conv2D (64 filters) -> BatchNorm -> ReLU -> MaxPool -> Dropout
3. **Fully Connected**: Flatten -> Dense (256 units) -> ReLU -> Dropout -> Dense (10 units)
## Results
- **Accuracy**: >98% on the test set.
- **Visualizations**:
    - `predictions.png`: Sample predictions vs ground truth.
    - `filters.png`: Visualization of the first convolutional layer filters.
    - `activations.png`: Feature maps from the first layer.
    - `training_history.png`: Loss and accuracy curves.
