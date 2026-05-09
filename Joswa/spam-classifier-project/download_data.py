import urllib.request
import os

# Create data folder
os.makedirs("data", exist_ok=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
save_path = "data/spambase.data"

urllib.request.urlretrieve(url, save_path)

print("Download complete! File saved to:", save_path)