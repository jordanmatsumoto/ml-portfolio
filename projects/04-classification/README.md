# Classification

## Description  
This project demonstrates **classification algorithms**, a supervised learning approach for predicting categorical outcomes. It applies seven classifiers to a social network ads dataset to predict whether a user will purchase a product based on **Age** and **Estimated Salary**.

## Techniques Covered
- **Logistic Regression**: Linear model for binary classification  
- **K-Nearest Neighbors (K-NN)**: Classifies points based on the majority class of nearest neighbors  
- **Support Vector Machine (SVM)**: Finds the optimal hyperplane for linear classification  
- **Kernel SVM**: Uses Gaussian/RBF kernel for non-linear decision boundaries  
- **Naive Bayes**: Probabilistic classifier assuming feature independence  
- **Decision Tree Classification**: Builds a tree structure based on feature splits using entropy  
- **Random Forest Classification**: Ensemble of decision trees for robust predictions  
- **Feature Scaling**: Standardization with `StandardScaler`  
- **Model Evaluation**: Confusion matrix and accuracy score  
- **Visualization**: Decision boundary plots for training and test sets  

## Dataset
- File: `data/Social_Network_Ads.csv`  
- Independent variable(s): Age, Estimated Salary  
- Dependent variable: Purchased (0 = No, 1 = Yes)  
- Source: Example dataset for classification tasks  

## Workflow
1. Load the dataset (`Social_Network_Ads.csv`) with Pandas  
2. Split the dataset into training and test sets (25% test size)  
3. Apply **feature scaling** using `StandardScaler`  
4. Train classifiers:  
   - Logistic Regression  
   - K-Nearest Neighbors (K-NN)  
   - Support Vector Machine (SVM)  
   - Kernel SVM  
   - Naive Bayes  
   - Decision Tree Classification  
   - Random Forest Classification  
5. Predict outcomes for new observations  
6. Evaluate models on the test set:  
   - Confusion matrix  
   - Accuracy score  
7. Visualize decision boundaries for training and test sets  

## How to Run
1. Clone the repository  
```bash
git clone https://github.com/jordanmatsumoto/ml-portfolio.git
```

2. Install dependencies from the root-level requirements.txt  
```bash
pip install -r ../../requirements.txt
```

3. Open the notebooks in Jupyter or VS Code and run cells sequentially:  
```bash
jupyter notebook 01_logistic_regression.ipynb
```
```bash
jupyter notebook 02_k_nearest_neighbors.ipynb
```
```bash
jupyter notebook 03_support_vector_machine.ipynb
```
```bash
jupyter notebook 04_kernel_svm.ipynb
```
```bash
jupyter notebook 05_naive_bayes.ipynb
```
```bash
jupyter notebook 06_decision_tree_classification.ipynb
```
```bash
jupyter notebook 07_random_forest_classification.ipynb
```

4. Follow the workflow in each notebook to reproduce model training, predictions, and visualizations

## Highlights / Results
- Models successfully predict user purchase behavior based on age and salary
- Decision boundaries visualize how each classifier separates classes
- Test set accuracy (approximate):
    - **Logistic Regression** ~89%
    - **K-Nearest Neighbors** ~91%
    - **Support Vector Machine** ~90%
    - **Kernel SVM** ~92%
    - **Naive Bayes** ~88%
    - **Decision Tree** ~86%
    - **Random Forest** ~92%

> **Note:** Actual accuracies may vary depending on train-test split and hyperparameters.

## File Structure
```
04-classification/
├── 01_logistic_regression.ipynb
├── 02_k_nearest_neighbors.ipynb
├── 03_support_vector_machine.ipynb
├── 04_kernel_svm.ipynb
├── 05_naive_bayes.ipynb
├── 06_decision_tree_classification.ipynb
├── 07_random_forest_classification.ipynb
├── README.md
└── data/
    └── Social_Network_Ads.csv
```