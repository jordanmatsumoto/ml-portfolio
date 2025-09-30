# Regression

## Description  
This project demonstrates **regression techniques** for predicting numerical outcomes. It applies six models — **Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, Support Vector Regression, Decision Tree Regression, and Random Forest Regression** — on three datasets to illustrate different regression strategies and model fitting.

## Techniques Covered
- **Simple Linear Regression**: Predicts a single dependent variable using one independent variable  
- **Multiple Linear Regression**: Predicts a dependent variable using multiple independent variables  
- **Polynomial Regression**: Captures non-linear relationships with polynomial features  
- **Support Vector Regression (SVR)**: Uses kernel methods to fit non-linear patterns  
- **Decision Tree Regression**: Fits stepwise predictions using a tree structure  
- **Random Forest Regression**: Ensemble of decision trees for smoother predictions  
- Feature scaling with **StandardScaler** where required  
- Model evaluation using **Test and Training R² scores**  
- Visualization of regression results and prediction curves  

## Dataset
- **Simple Linear Regression:**  
  - File: `data/Salary_Data.csv`  
  - Independent variable(s): Position level  
  - Dependent variable: Salary  
  - Source: Example dataset for simple linear regression  

- **Multiple Linear Regression:**  
  - File: `data/50_Startups.csv`  
  - Independent variable(s): R&D Spend, Administration, Marketing Spend, State (dummy variables)  
  - Dependent variable: Profit  
  - Source: Example dataset for multiple linear regression  

- **Polynomial, SVR, Decision Tree & Random Forest Regression:**  
  - File: `data/Position_Salaries.csv`  
  - Independent variable(s): Position level  
  - Dependent variable: Salary  
  - Source: Example dataset for polynomial regression, SVR, decision tree regression, and random forest regression  

## Workflow
1. Load the dataset with Pandas  
2. Preprocess data: feature selection, encoding categorical variables, or feature scaling  
3. Train the regression model:  
  - **Linear and Multiple Linear Regression** → fit with `LinearRegression()`  
  - **Polynomial Regression** → transform features with `PolynomialFeatures(degree=4)`  
  - **SVR** → scale features with `StandardScaler` and fit with `SVR(kernel='rbf')`  
  - **Decision Tree Regression** → fit with `DecisionTreeRegressor(random_state=0)`  
  - **Random Forest Regression** → fit with `RandomForestRegressor(n_estimators=10, random_state=0)`  
4. Predict new results and visualize predictions  
5. Calculate **R² scores** to evaluate fit  

## How to Run
1. Clone the repository  
```bash
git clone https://github.com/jordanmatsumoto/ml-portfolio.git
```

2. Install all dependencies from the root-level requirements.txt  
```bash
pip install -r ../../requirements.txt
```

3. Open the notebooks in Jupyter or VS Code  
```bash
jupyter notebook 01_simple_linear_regression.ipynb
```
```bash
jupyter notebook 02_multiple_linear_regression.ipynb
```
```bash
jupyter notebook 03_polynomial_regression.ipynb
```
```bash
jupyter notebook 04_support_vector_regression.ipynb
```
```bash
jupyter notebook 05_decision_tree_regression.ipynb
```
```bash
jupyter notebook 06_random_forest_regression.ipynb
```

4. Run cells sequentially to reproduce all regression models, plots, and R² scores

## Highlights / Results
- **Simple Linear Regression**: Baseline linear fit. Test R² reported.  
- **Multiple Linear Regression**: Multiple features predicting profit. Test R² reported.  
- **Polynomial Regression**: Fits non-linear trends. Training R² reported.  
- **Support Vector Regression (SVR)**: Smooth curve fitting. Training R² reported.  
- **Decision Tree Regression**: Stepwise predictions. Training R² ≈ 1.  
- **Random Forest Regression**: Ensemble of trees. Training R² ≈ 1.  

> **Note:** R² values are reported as Training R² for models trained on the full dataset, and Test R² for models with a train/test split. These indicate model fit for each respective dataset

## File Structure
```
03-regression/
├── 01_simple_linear_regression.ipynb
├── 02_multiple_linear_regression.ipynb
├── 03_polynomial_regression.ipynb
├── 04_support_vector_regression.ipynb
├── 05_decision_tree_regression.ipynb
├── 06_random_forest_regression.ipynb
├── README.md
└── data/
  ├── Salary_Data.csv
  ├── 50_Startups.csv
  └── Position_Salaries.csv
```