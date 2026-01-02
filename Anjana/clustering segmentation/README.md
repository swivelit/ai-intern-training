
# Customer Segmentation using Clustering

## Dataset
Mall Customers Dataset (Kaggle):
https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial

Download `Mall_Customers.csv` and place it in the `data/` folder.

## Methods Used
- K-Means
- Hierarchical (Agglomerative)
- DBSCAN

## Features
- Annual Income (k$)
- Spending Score (1-100)

## Business Interpretation (Summary)
- **High Income, High Spending**: Premium customers – target with loyalty and premium offers.
- **High Income, Low Spending**: Potential customers – focus on personalized promotions.
- **Low Income, High Spending**: Deal seekers – target with discounts.
- **Low Income, Low Spending**: Low priority – minimal marketing spend.

## How to Run
```bash
pip install -r requirements.txt
python src/clustering.py
```
Plots will be saved in the `plots/` folder.
