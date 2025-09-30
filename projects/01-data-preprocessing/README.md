# Data Preprocessing

## Description  
This project demonstrates **essential data preprocessing techniques** for preparing datasets in machine learning workflows. It covers handling missing values, encoding categorical data, splitting into training and testing sets, and applying feature scaling.

## Techniques Covered
- Importing libraries: `numpy`, `pandas`, `matplotlib`
- Handling missing data using `SimpleImputer` (mean strategy)
- Encoding categorical variables:
  - One-Hot Encoding for independent features
  - Label Encoding for dependent variable
- Splitting the dataset into training and testing sets (`train_test_split`)
- Feature scaling using `StandardScaler`  

## Dataset
- File: `data/Data.csv`
- Independent variable(s): Columns 0 to n-2 (X)
- Dependent variable: Last column (y)
- Source: Example dataset for data preprocessing

## Workflow
1. Load the dataset using `pandas`
2. Separate features (`X`) and target (`y`)
3. Handle missing values in columns 1 and 2 using mean imputation
4. Encode categorical features:
   - Independent variable column 0 → One-Hot Encoding
   - Dependent variable → Label Encoding
5. Split dataset into **training** and **test** sets (80/20 split)
6. Apply **feature scaling** to numerical features (columns 3 onward)
7. Print intermediate results for verification

## How to Run
1. Clone the repository  
```bash
git clone https://github.com/jordanmatsumoto/ml-portfolio.git
```

2. Install all dependencies from the root-level requirements.txt  
```bash
pip install -r ../../requirements.txt
```

3. Open the notebook in Jupyter or VS Code  
```bash
jupyter notebook 01_data_preprocessing_tools.ipynb
```

4. Run cells sequentially to reproduce the preprocessing workflow

## Highlights / Results
- Successfully handled missing data in the dataset
- Categorical data properly encoded for machine learning algorithms
- Dataset split into training and testing sets
- Numerical features standardized using feature scaling
- Data is now fully ready for modeling with regression, classification, or other ML algorithms

## File Structure
```
01-data-preprocessing/
├── 01_data_preprocessing_tools.ipynb
├── README.md
└── data/
    └── Data.csv
```