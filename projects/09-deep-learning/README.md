# Deep Learning

## Description
This project demonstrates **deep learning** techniques, including **Artificial Neural Networks (ANN)** for classification and regression, and a **Convolutional Neural Network (CNN)** for image classification (cats vs. dogs). It covers data preprocessing, building and training neural networks, evaluating models, and making predictions.

## Techniques Covered
- **Artificial Neural Networks (Classification & Regression):**
  - Feature scaling, encoding categorical variables (classification only)  
  - ANN architecture with multiple hidden layers  
  - Activation functions: ReLU for hidden layers, Sigmoid (classification) / Linear (regression) for output  
  - Model evaluation: Accuracy (classification) and comparison of predicted vs actual values (regression)  
  - Predicting single observations (classification)  

- **Convolutional Neural Network (CNN):**
  - Data preprocessing: CIFAR-10 dataset filtered for cats and dogs, normalization  
  - CNN architecture with convolutional, pooling, flattening, and dense layers  
  - Activation functions: ReLU for hidden layers, Sigmoid for output  
  - Model evaluation: Accuracy on full test set  
  - Predictions and visualization of multiple sample images  

## Dataset
- **ANN Classification:**
  - File: `data/Churn_Modelling.csv`  
  - Independent variable(s): Credit Score, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary  
  - Dependent variable: Exited (Customer churn: 0 = no, 1 = yes)  
  - Source: Example dataset for ANN classification  

- **ANN Regression:** 
  - File: `data/Folds5x2_pp.xlsx`  
  - Independent variable(s): All features in the dataset (e.g., temperature, pressure, humidity, etc.)  
  - Dependent variable: CO (carbon monoxide concentration) – continuous variable  
  - Source: Example dataset for ANN regression  

- **CNN:**
  - File: CIFAR-10 (built-in)  
  - Independent variable(s): Pixel values of 32×32 RGB images  
  - Dependent variable: Class labels – cats vs dogs  
  - Source: Example dataset for convolutional neural networks (CNNs); no local file required

## Workflow
### ANN Classification
1. Load the dataset and encode categorical features (gender, geography)  
2. Split into training and test sets, apply feature scaling  
3. Build the ANN with two hidden layers (6 units each)  
4. Compile the ANN (optimizer = Adam, loss = binary_crossentropy)  
5. Train the ANN on the training set  
6. Evaluate the model on the test set using accuracy and confusion matrix  
7. Predict the outcome of a single customer  

### ANN Regression
1. Load the dataset and split into training and test sets  
2. Build the ANN with two hidden layers (6 units each)  
3. Compile the ANN (optimizer = Adam, loss = mean_squared_error)  
4. Train the ANN on the training set  
5. Evaluate predictions against actual test values  

### CNN
1. Load CIFAR-10 dataset, filter for cats (0) and dogs (1)  
2. Normalize images (divide pixel values by 255)  
3. Build the CNN with:
   - Two convolutional + pooling layers  
   - Flatten layer  
   - Dense layer (128 units, ReLU)  
   - Output layer (1 unit, Sigmoid)  
4. Compile the CNN (optimizer = Adam, loss = binary_crossentropy, metrics = accuracy)  
5. Train the CNN on the training set with validation on the test set  
6. Evaluate the CNN on the full test set  
7. Predict and visualize multiple sample images from the test set  
   - Change `num_samples` to test different images  

## How to Run
1. Clone the repository  
```bash
git clone https://github.com/jordanmatsumoto/ml-portfolio.git
```
2. Install dependencies from the root-level requirements.txt  
```bash
pip install -r ../../requirements.txt
```
3. Open the notebooks in Jupyter or Google Colab  
```bash
jupyter notebook 01_artificial_neural_network.ipynb
```
```bash
jupyter notebook 02_artificial_neural_network_regression.ipynb
```
```bash
jupyter notebook 03_convolutional_neural_network.ipynb
```
4. Run cells sequentially to reproduce preprocessing, model training, evaluation, and predictions

## Highlights / Results
- **ANN Classification**: Predicts customer churn accurately; single-observation predictions included
- **ANN Regression**: Predicts continuous values; test set results printed alongside actual values
- **CNN**: Classifies cats vs dogs; evaluates full test set; visualizes multiple predictions
- Models and workflows are portfolio-ready, with clear structure, evaluation, and visualization

## File Structure
```
09-deep-learning/
├── 01_artificial_neural_network.ipynb
├── 02_artificial_neural_network_regression.ipynb
├── 03_convolutional_neural_network.ipynb
├── README.md
└── data/
    ├── Churn_Modelling.csv
    └── Folds5x2_pp.xlsx
```