# NLP Coursework 2026: PCL Detection

**Student:** Safeyah Alshemali  
**CID:** 06032100  
**Leaderboard Name:** Hakuna_Matata

## Results
- Dev F1: 0.612 (baseline: 0.48, 27.5% improvement)
- Approach: RoBERTa-base + + Clean Data from HTML tag + class-weighted loss + train/val split + threshold tuning

## Repository Structure
- `EDA.py` - exploring data analysis 
- `BestModel/` — trained model weights and tokenizer
- `dev.txt` — predictions on official dev set (2,093 lines)
- `test.txt` — predictions on official test set (3,832 lines)
- `NLPCoursework_FINAL.ipynb` — full training and evaluation notebook
