# Natural Language Processing

## Description  
This project demonstrates **Natural Language Processing (NLP)** for **sentiment analysis** of restaurant reviews. The workflow includes preprocessing text data, training a machine learning model, evaluating its performance, and predicting the sentiment of new reviews (positive or negative).

## Techniques Covered
- **Text Preprocessing:** Removing non-letter characters, converting to lowercase, tokenization, removing stopwords (except “not”), and stemming using PorterStemmer  
- **Bag of Words Model:** Feature extraction using `CountVectorizer`  
- **Naive Bayes Classifier:** Gaussian Naive Bayes for sentiment classification  
- **Model Evaluation:** Confusion matrix and accuracy score  
- **Prediction on New Data:** Predicting sentiment of single reviews (both positive and negative examples)  

## Dataset
- File: `data/Restaurant_Reviews.tsv`  
- Independent variable(s): Text reviews of restaurants  
- Dependent variable: Sentiment label (1 = positive, 0 = negative)  
- Source: Example dataset for NLP sentiment analysis  

## Workflow
1. Load the dataset (`Restaurant_Reviews.tsv`) with Pandas  
2. Clean the texts:  
   - Remove non-letter characters  
   - Convert to lowercase  
   - Tokenize and remove stopwords (except “not”)  
   - Apply stemming  
3. Create the Bag of Words model using `CountVectorizer`  
4. Split the dataset into training and test sets  
5. Train a **Gaussian Naive Bayes** classifier on the training set  
6. Evaluate the model using test set predictions:  
   - Compute confusion matrix  
   - Calculate accuracy score  
7. Predict the sentiment of new reviews using the trained model:  
   - Positive example: “I love this restaurant so much” → **Positive**  
   - Negative example: “I hate this restaurant so much” → **Negative**  

## How to Run
1. Clone the repository  
```bash
git clone https://github.com/jordanmatsumoto/ml-portfolio.git
```
2. Install dependencies from the root-level requirements.txt  
```bash
pip install -r ../../requirements.txt
```
3. Open the notebook in Jupyter or VS Code  
```bash
jupyter notebook 01_natural_language_processing.ipynb
```
4. Run cells sequentially to reproduce the NLP workflow, including predicting sentiment of new reviews

## Highlights / Results
- Preprocessing successfully converts raw text into a feature matrix suitable for machine learning
- Gaussian Naive Bayes achieves accurate sentiment classification on test data
- The model can predict sentiment of new reviews, demonstrating real-world usability:
    - Positive example → Positive
    - Negative example → Negative

## File Structure
```
08-natural-language-processing/
├── 01_natural_language_processing.ipynb
├── README.md
└── data/
   └── Restaurant_Reviews.tsv
```