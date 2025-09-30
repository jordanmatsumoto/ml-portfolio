# Dimensionality Reduction

## Description  
This project demonstrates **dimensionality reduction techniques** for reducing feature space while preserving key information in classification tasks. It applies **PCA, LDA, and Kernel PCA** to the **Wine dataset**, then trains Logistic Regression models to evaluate classification performance.

## Techniques Covered
- **Principal Component Analysis (PCA)**: Linear transformation to reduce features while maximizing variance  
- **Linear Discriminant Analysis (LDA)**: Supervised method maximizing class separability  
- **Kernel PCA (KPCA)**: Nonlinear dimensionality reduction using the RBF kernel  
- **StandardScaler** for feature scaling  
- Logistic Regression for classification  
- Model evaluation with **Confusion Matrix** and **Accuracy Score**  
- Visualization of decision boundaries in 2D  

## Dataset
- File: `data/Wine.csv`  
- Independent variable(s): physicochemical properties of wines  
- Dependent variable: wine class label (1–3)  
- Source: Example dataset for classification tasks  

## Workflow
1. Load the dataset (`Wine.csv`) with Pandas  
2. Standardize features with `StandardScaler`  
3. Apply dimensionality reduction:  
   - **PCA** → extract 2 principal components  
   - **LDA** → extract 2 linear discriminants  
   - **KPCA** → extract 2 kernel-based components (RBF kernel)  
4. Train **Logistic Regression** on reduced datasets  
5. Evaluate with **confusion matrix** and **accuracy score**  
6. Visualize classification boundaries for training and test sets  

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
jupyter notebook 01_principal_component_analysis.ipynb
```
```bash
jupyter notebook 02_linear_discriminant_analysis.ipynb
```
```bash
jupyter notebook 03_kernel_pca.ipynb
```

4. Run cells sequentially to reproduce the dimensionality reduction workflow

## Highlights / Results
- **PCA**: Demonstrates separation of classes by variance maximization  
- **LDA**: Achieves stronger class separation due to supervised learning  
- **KPCA**: Captures nonlinear patterns for potentially better separation  

## File Structure
```
02-dimensionality-reduction/  
├── 01_principal_component_analysis.ipynb  
├── 02_linear_discriminant_analysis.ipynb  
├── 03_kernel_pca.ipynb 
├── README.md  
└── data/  
    └── Wine.csv
```