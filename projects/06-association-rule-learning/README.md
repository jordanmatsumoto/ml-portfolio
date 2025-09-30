# Association Rule Learning

## Description  
This project demonstrates **association rule learning**, a data mining technique for discovering relationships between items in large transactional datasets. It applies **Apriori** and **Eclat** algorithms to a market basket dataset to uncover patterns in customer purchases.

## Techniques Covered
- **Apriori Algorithm**: Finds frequent itemsets and generates association rules based on minimum support, confidence, and lift  
- **Eclat Algorithm**: Finds frequent itemsets using a depth-first search and intersection approach for fast computation  
- **Data Preprocessing**: Transforming transactional data into a format suitable for association rule mining  
- **Rule Evaluation Metrics**: Support, Confidence, Lift  

## Dataset
- File: `data/Market_Basket_Optimization.csv`  
- Independent variable(s): Purchased items per transaction (each column represents a product; empty if not purchased) 
- Dependent variable: None (unsupervised association rule learning)    
- Source: Example market basket dataset  

## Workflow
1. Load the dataset (`Market_Basket_Optimization.csv`) with Pandas  
2. Transform dataset into a list of transactions  
3. **Apriori:**  
   - Generate frequent itemsets and association rules  
   - Filter rules by minimum support, confidence, and lift  
   - Visualize and sort rules by lift  
4. **Eclat:**  
   - Generate frequent itemsets  
   - Filter rules by minimum support  
   - Visualize and sort rules by support  

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
jupyter notebook 01_apriori.ipynb
```
```bash
jupyter notebook 02_eclat.ipynb
```
4. Run cells sequentially to reproduce the association rule mining workflow

## Highlights / Results
- **Apriori Algorithm**: Identified top association rules with high lift, highlighting strong relationships between products
- **Eclat Algorithm**: Extracted frequent itemsets efficiently, sorted by support to reveal the most commonly purchased product pairs
- Insights can inform product bundling, cross-selling strategies, and marketing campaigns

## File Structure
```
06-association-rule-learning/
├── 01_apriori.ipynb
├── 02_eclat.ipynb
├── README.md
└── data/
    └── Market_Basket_Optimisation.csv
```