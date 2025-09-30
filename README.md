# ML Portfolio

This repository contains a collection of **Machine Learning projects** demonstrating a wide range of techniques, from data preprocessing to deep learning and boosting. Each project includes datasets, notebooks, visualizations, and explanations of workflows, providing a comprehensive showcase of ML skills.

## Projects Overview
1. **Data Preprocessing** – Cleaning, encoding, scaling, and preparing datasets for ML  
2. **Dimensionality Reduction** – PCA, LDA, Kernel PCA for feature reduction  
3. **Regression** – Linear, Polynomial, SVR, Decision Tree, Random Forest  
4. **Classification** – Logistic Regression, K-NN, SVM, Kernel SVM, Naive Bayes, Decision Tree, Random Forest  
5. **Clustering** – K-Means and Hierarchical Clustering  
6. **Association Rule Learning** – Apriori and Eclat algorithms for market basket analysis  
7. **Reinforcement Learning** – Upper Confidence Bound (UCB) and Thompson Sampling  
8. **Natural Language Processing** – Sentiment analysis with Naive Bayes  
9. **Deep Learning** – ANN for classification/regression, CNN for image classification  
10. **Model Selection & Boosting** – Grid Search, K-Fold Cross-Validation, XGBoost, CatBoost  

## 01 Data Preprocessing
- **Description:** Essential preprocessing techniques (missing data, encoding, feature scaling, train/test split)  
- **Dataset:** `Data.csv` (Independent variable(s): Columns 0 to n-2 (X); Dependent variable: Last column (y))  
- **Notebook:** `01_data_preprocessing_tools.ipynb`  
- **Highlights:** Dataset cleaned, encoded, scaled, and ready for modeling  

## 02 Dimensionality Reduction
- **Description:** Reduce feature space while preserving information using PCA, LDA, Kernel PCA  
- **Dataset:** `Wine.csv` (Independent variable(s): physicochemical properties of wines; Dependent variable: wine class label (1–3))  
- **Notebooks:** `01_principal_component_analysis.ipynb`, `02_linear_discriminant_analysis.ipynb`, `03_kernel_pca.ipynb`  
- **Highlights:** PCA shows variance-based separation; LDA shows class separability; KPCA captures nonlinear patterns  

## 03 Regression
- **Description:** Predict numerical outcomes using multiple regression techniques  
- **Datasets:** 
  - `Salary_Data.csv` – Simple Linear Regression (Independent variable(s): Position level; Dependent variable: Salary)
  - `50_Startups.csv` – Multiple Linear Regression (Independent variable(s): R&D Spend, Administration, Marketing Spend, State (dummy variables); Dependent variable: Profit)
  - `Position_Salaries.csv` – Polynomial Regression, SVR, Decision Tree, Random Forest (Independent variable(s): Position level; Dependent variable: Salary)  
- **Notebooks:** `01_simple_linear_regression.ipynb` … `06_random_forest_regression.ipynb`  
- **Highlights:** Models fit data well; R² values reported for training/test sets; predictions visualized  

## 04 Classification
- **Description:** Predict categorical outcomes using 7 classifiers  
- **Dataset:** `Social_Network_Ads.csv` (Independent variable(s): Age, Estimated Salary; Dependent variable: Purchased (0 = No, 1 = Yes))   
- **Notebooks:** `01_logistic_regression.ipynb` … `07_random_forest_classification.ipynb`  
- **Highlights:** Decision boundaries visualized; test set accuracies estimated:  
  - **Logistic Regression** ~89%  
  - **K-Nearest Neighbors** ~91%  
  - **SVM** ~90%  
  - **Kernel SVM** ~92%  
  - **Naive Bayes** ~88%  
  - **Decision Tree** ~86%  
  - **Random Forest** ~92%  

## 05 Clustering
- **Description:** Group similar data points without labels  
- **Dataset:** `Mall_Customers.csv` (Independent variable(s): Annual Income (k$), Spending Score (1-100); Dependent variable: None (unsupervised clustering))  
- **Notebooks:** `01_k_means_clustering.ipynb`, `02_hierarchical_clustering.ipynb`  
- **Highlights:** 5 distinct clusters found; dendrogram visualizes hierarchical clustering  

## 06 Association Rule Learning
- **Description:** Discover item relationships in transactions  
- **Dataset:** `Market_Basket_Optimisation.csv` (Independent variable(s): Purchased items per transaction (each column represents a product; empty if not purchased); Dependent variable: None (unsupervised association rule learning))  
- **Notebooks:** `01_apriori.ipynb`, `02_eclat.ipynb`  
- **Highlights:** Apriori identifies high-lift rules; Eclat extracts frequent itemsets efficiently  

## 07 Reinforcement Learning
- **Description:** Optimize ad selection for maximum CTR  
- **Dataset:** `Ads_CTR_Optimisation.csv` (Independent variable(s): Ads (10 columns, one per ad); Dependent variable: Click (1 = clicked, 0 = not clicked) over 10,000 rounds)  
- **Notebooks:** `01_upper_confidence_bound.ipynb`, `02_thompson_sampling.ipynb`  
- **Highlights:** UCB balances exploration/exploitation; Thompson Sampling adapts probabilistically; histogram visualizations  

## 08 Natural Language Processing
- **Description:** Sentiment analysis of restaurant reviews  
- **Dataset:** `Restaurant_Reviews.tsv` (Independent variable(s): Text reviews of restaurants; Dependent variable: Sentiment label (1 = positive, 0 = negative)) 
- **Notebook:** `01_natural_language_processing.ipynb`  
- **Highlights:** Text preprocessing successful; Gaussian Naive Bayes predicts positive/negative sentiment accurately  

## 09 Deep Learning
- **Description:** ANN (classification/regression) and CNN (cats vs dogs) projects  
- **Datasets:**
  - `Churn_Modelling.csv` - ANN classification (Independent variable(s): Credit Score, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary; Dependent variable: Exited)  
  - `Folds5x2_pp.xlsx` - ANN regression (Independent variable(s): All features in the dataset (e.g., temperature, pressure, humidity, etc.); Dependent variable: CO (carbon monoxide concentration) – continuous variable) 
  - CIFAR-10 - CNN (Independent variable(s): Pixel values of 32×32 RGB images; Dependent variable: Class labels – cats vs dogs)  
- **Notebooks:** `01_artificial_neural_network.ipynb`, `02_artificial_neural_network_regression.ipynb`, `03_convolutional_neural_network.ipynb`  
- **Highlights:** ANN predicts churn and continuous values; CNN classifies cats/dogs; workflows fully reproducible  

## 10 Model Selection & Boosting
- **Description:** Hyperparameter tuning, cross-validation, XGBoost, and CatBoost  
- **Datasets:** 
  - `Social_Network_Ads.csv` - Grid Search & K-Fold Cross-Validation (Independent variable(s): Age, Estimated Salary; Dependent variable: Purchased)  
  - `Data.csv` - XGBoost & CatBoost (Independent variable(s): Columns 0 to n-2 (X); Dependent variable: Last column (y))  
- **Notebooks:** `01_grid_search.ipynb`, `02_k_fold_cross_validation.ipynb`, `03_xgboost.ipynb`, `04_catboost.ipynb`  
- **Highlights:** Optimized models improve performance; XGBoost and CatBoost achieve high accuracy; workflows enable model comparison  

## How to Run the Full Portfolio
1. Clone the repository  
```bash
git clone https://github.com/jordanmatsumoto/ml-portfolio.git
```
    
2. Install dependencies
```bash  
pip install -r requirements.txt
```

3. Open any notebook in Jupyter or VS Code and run cells sequentially to reproduce results

## File Structure
```
ml-portfolio/
├── projects/
│   ├── 01-data-preprocessing/
│   ├── 02-dimensionality-reduction/
│   ├── 03-regression/
│   ├── 04-classification/
│   ├── 05-clustering/
│   ├── 06-association-rule-learning/
│   ├── 07-reinforcement-learning/
│   ├── 08-natural-language-processing/
│   ├── 09-deep-learning/
│   └── 10-model-selection-boosting/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```