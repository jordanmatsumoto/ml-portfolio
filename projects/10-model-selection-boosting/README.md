# Model Selection & Boosting

## Description
This project demonstrates **model selection techniques** and **boosting algorithms** for supervised machine learning. It explores **grid search** and **k-fold cross-validation** to optimize hyperparameters and evaluate model stability, then implements **XGBoost** and **CatBoost** for predictive modeling on a separate dataset.

## Techniques Covered
- **Grid Search:** Exhaustive search over specified hyperparameter values to optimize model performance  
- **K-Fold Cross-Validation:** Splitting the training set into k folds to evaluate model stability and prevent overfitting  
- **XGBoost:** Gradient boosting framework optimized for speed and accuracy  
- **CatBoost:** Gradient boosting with categorical feature support and ordered boosting to reduce overfitting  
- **Model Evaluation:** Accuracy, confusion matrix, and k-fold cross-validation results  

## Dataset
- **Grid Search & K-Fold Cross-Validation:**  
  - File: `data/Social_Network_Ads.csv`  
  - Independent variable(s): Age, Estimated Salary  
  - Dependent variable: Purchased (0 = No, 1 = Yes)  
  - Source: Example dataset for hyperparameter tuning and cross-validation  

- **XGBoost & CatBoost:**  
  - File: `data/Data.csv`  
  - Independent variable(s): Columns 0 to n-2 (X)
  - Dependent variable: Last column (y)
  - Source: Example dataset for demonstrating gradient boosting algorithms  

## Workflow
1. Load the appropriate dataset with Pandas  
2. **Model Selection:**  
   - Apply **Grid Search** to find optimal hyperparameters for a Kernel SVM model  
   - Apply **K-Fold Cross-Validation** to evaluate model stability  
3. **Boosting Algorithms:**  
   - Train an **XGBoost** model on the boosting dataset  
   - Train a **CatBoost** model on the boosting dataset  
4. Evaluate model performance using metrics:  
   - Confusion matrix  
   - Accuracy score  
   - K-Fold cross-validation (mean accuracy and standard deviation)  
5. Compare results to identify the best-performing approach  

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
jupyter notebook 01_grid_search.ipynb
```
```bash
jupyter notebook 02_k_fold_cross_validation.ipynb
```
```bash
jupyter notebook 03_xgboost.ipynb
```
```bash
jupyter notebook 04_catboost.ipynb
```
4. Run cells sequentially to reproduce the model selection and boosting workflow

## Highlights / Results
- **Grid Search & K-Fold**: Optimized hyperparameters improve model performance and stability
- **XGBoost**: Fast, accurate predictions demonstrating gradient boosting power
- **CatBoost**: Efficient handling of categorical features with high predictive accuracy
- Enables comparison of model selection strategies and boosting algorithms for informed decision-making

## File Structure
```
10-model-selection-boosting/
├── 01_grid_search.ipynb
├── 02_k_fold_cross_validation.ipynb
├── 03_xgboost.ipynb
├── 04_catboost.ipynb
├── README.md
└── data/
    ├── Social_Network_Ads.csv
    └── Data.csv
```