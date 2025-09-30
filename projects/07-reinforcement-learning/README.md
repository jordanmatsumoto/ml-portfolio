# Reinforcement Learning

## Description  
This project demonstrates **Reinforcement Learning (RL) algorithms** applied to online ad selection, aiming to maximize click-through rate (CTR) over time. It implements **Upper Confidence Bound (UCB)** and **Thompson Sampling** to learn which ads perform best while balancing exploration and exploitation.

## Techniques Covered
- **Upper Confidence Bound (UCB):** Selects actions (ads) by maximizing an upper confidence bound for expected reward  
- **Thompson Sampling:** Probabilistic approach using Beta distributions to balance exploration and exploitation  
- Histogram visualization of ad selections over time  
- Use of cumulative rewards to evaluate algorithm performance  

## Dataset
- File: `data/Ads_CTR_Optimisation.csv`  
- Independent variable(s): Ads (10 columns, one per ad)  
- Dependent variable: Click (1 = clicked, 0 = not clicked) over 10,000 rounds  
- Source: Example CTR optimization dataset  

## Workflow
1. Load the dataset (`Ads_CTR_Optimisation.csv`) with Pandas  
2. **Upper Confidence Bound (UCB):**  
   - Initialize counters and rewards for each ad  
   - Iteratively select ads based on upper confidence bound  
   - Update counters and cumulative rewards  
   - Visualize the number of selections per ad using a histogram  
3. **Thompson Sampling:**  
   - Initialize success/failure counts for each ad  
   - Iteratively sample from Beta distributions to select ads  
   - Update success/failure counts and cumulative rewards  
   - Visualize the number of selections per ad using a histogram  

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
jupyter notebook 01_upper_confidence_bound.ipynb
```
```bash
jupyter notebook 02_thompson_sampling.ipynb
```
4. Run cells sequentially to reproduce the reinforcement learning workflow

## Highlights / Results
- **Upper Confidence Bound (UCB)**: Efficiently balances exploration and exploitation, selecting the best performing ads more frequently over time
- **Thompson Sampling**: Uses probabilistic sampling to adaptively select ads, often achieving similar or better cumulative rewards than UCB
- Histograms provide a clear visual representation of how often each ad was selected, demonstrating algorithm behavior

## File Structure
```
07-reinforcement-learning/
├── 01_upper_confidence_bound.ipynb
├── 02_thompson_sampling.ipynb
├── README.md
└── data/
    └── Ads_CTR_Optimisation.csv
```