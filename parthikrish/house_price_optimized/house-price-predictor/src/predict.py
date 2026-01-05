import joblib, numpy as np

lr = joblib.load("models/lr.pkl")
dt = joblib.load("models/dt.pkl")
rf = joblib.load("models/rf.pkl")
pr = joblib.load("models/pr.pkl")
poly = joblib.load("models/poly.pkl")

# Sample input (8 features for Boston, adjust for California if used)
sample = np.random.rand(1, lr.n_features_in_)

print("Linear Prediction:", lr.predict(sample))
print("Tree Prediction:", dt.predict(sample))
print("Forest Prediction:", rf.predict(sample))
print("Poly Prediction:", pr.predict(poly.transform(sample)))
