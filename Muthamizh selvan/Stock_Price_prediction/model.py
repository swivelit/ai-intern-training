from sklearn.neural_network import MLPRegressor

def build_mlp_model():
    """
    Builds a Multi-Layer Perceptron for regression using scikit-learn.
    """
    # hidden_layer_sizes=(64, 32, 16) matches the architecture previously defined
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    return model
