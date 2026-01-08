# Customer Segmentation using Clustering

## 📊 Project Overview
This project segments mall customers based on their annual income and spending score using clustering techniques. The goal is to identify customer groups for targeted marketing strategies.

## 📁 Dataset
Mall Customers Dataset  
Source: Kaggle  
Features used:
- Annual Income (k$)
- Spending Score (1–100)

## 🧠 Algorithms Used
1. K-Means Clustering
2. Hierarchical Clustering
3. DBSCAN

## 📈 Visualizations
- Elbow Method (K-Means)
- Cluster Scatter Plots
- Hierarchical Dendrogram
- DBSCAN clusters with noise detection

## 🧩 Business Interpretation
- High Income + High Spending → Premium customers
- High Income + Low Spending → Potential customers
- Low Income + High Spending → Deal seekers
- Low Income + Low Spending → Low priority
- DBSCAN Noise → Unusual or outlier behavior

## 🛠 Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn, SciPy

## ▶ How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run the script:
   python customer_segmentation.py

## ✅ Output
- Visual customer segments
- Business-ready insights
