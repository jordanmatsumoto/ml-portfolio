# Clustering

## Description  
This project demonstrates **clustering techniques** for grouping similar data points without labeled outputs. It applies **K-Means Clustering** and **Hierarchical Clustering** to segment customers based on annual income and spending score.

## Techniques Covered
- **K-Means Clustering**: Partition data into clusters using centroids and distance minimization  
- **Elbow Method**: Determine the optimal number of clusters by analyzing WCSS (within-cluster sum of squares)  
- **Hierarchical Clustering (Agglomerative)**: Build a hierarchy of clusters using Euclidean distance and linkage  
- **Dendrograms**: Visual tool to determine the optimal number of clusters  
- Visualization of clusters in 2D  

## Dataset
- File: `data/Mall_Customers.csv`  
- Independent variable(s): Annual Income (k$), Spending Score (1-100)  
- Dependent variable: None (unsupervised clustering)  
- Source: Example dataset for customer segmentation  

## Workflow
1. Load the dataset (`Mall_Customers.csv`) with Pandas  
2. Extract relevant features for clustering  
3. **K-Means:** Use the Elbow Method to find the optimal number of clusters, train the model, and visualize clusters  
4. **Hierarchical Clustering:** Use a dendrogram to determine the number of clusters, train the model, and visualize clusters  

## How to Run
1. Clone the repository  
```bash
git clone https://github.com/jordanmatsumoto/ml-portfolio.git
```
2. Install dependencies from the root-level requirements.txt  
```bash
pip install -r ../../requirements.txt
```
3. Open the notebooks in Jupyter or VS Code  
```bash
jupyter notebook 01_k_means_clustering.ipynb
```
```bash
jupyter notebook 02_hierarchical_clustering.ipynb
```
4. Run cells sequentially to reproduce the clustering workflow

## Highlights / Results
- **K-Means Clustering**: Customers segmented into 5 distinct clusters. Centroids highlight average positions of each cluster.
- **Hierarchical Clustering**: Agglomerative clustering produces similar 5-cluster segmentation. Dendrogram shows cluster distances and hierarchy.
- Visualizations make cluster separations clear for marketing segmentation or targeted campaigns.

## File Structure
```
05-clustering/
├── 01_k_means_clustering.ipynb
├── 02_hierarchical_clustering.ipynb
├── README.md
└── data/
    └── Mall_Customers.csv
```