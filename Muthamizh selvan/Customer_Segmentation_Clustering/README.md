# Customer Segmentation using Clustering

This project explores unsupervised learning techniques to segment customers of a mall based on their annual income and spending score.

## Dataset
- **Source**: Mall Customers Dataset (Kaggle)
- **Features Used**: Annual Income (k$), Spending Score (1-100)

## Implemented Algorithms
1.  **K-Means**: Efficiently partitions data into 5 distinct groups.
2.  **Hierarchical (Agglomerative)**: Uses a ward-linkage method to find structural clusters.
3.  **DBSCAN**: Density-based clustering that can identify noise (outliers).

## How to Run
1.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the main script:
    ```bash
    python main.py
    ```

## Outputs
- **Plots**: Visualizations for each algorithm are saved in the `plots/` directory.
- **Dendrogram**: A tree-like diagram showing the hierarchical merging of clusters.
- **Interpretation**: A business summary of each customer segment is printed in the console.

## Business Insights
- **Standard Customers**: Mid-range income and spending.
- **VIP/Target**: High income and high spending (Maximize loyalty).
- **Careful**: High income but low spending (Incentivize with luxury offers).
- **Careless**: Low income but high spending (Target with budget-friendly high-volume offers).
- **Sensible**: Low income and low spending (Hard to target, focus on essential deals).
