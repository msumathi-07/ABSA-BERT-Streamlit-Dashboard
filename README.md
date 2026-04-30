# ABSA-BERT-Streamlit-Dashboard

## Project Overview
This project implements an Aspect-Based Sentiment Analysis (ABSA) system that analyzes customer reviews at a granular level by identifying product aspects and predicting sentiment for each aspect.

The system combines natural language processing techniques for aspect extraction with a fine-tuned DistilBERT model for sentiment classification. The results are presented through an interactive Streamlit dashboard.

---

## Features
- Extracts product aspects using SpaCy dependency parsing
- Performs sentiment classification using a fine-tuned DistilBERT model
- Handles multiple aspects within a single review
- Supports both single review input and batch analysis using CSV files
- Provides visual insights including:
  - Sentiment distribution
  - Aspect-level performance
  - Net Promoter Score (NPS)
  - Radar chart for strengths and weaknesses

---

## Tech Stack
- Python
- Streamlit
- SpaCy
- Hugging Face Transformers (DistilBERT)
- PyTorch
- Pandas
- Matplotlib
- Seaborn

---

## Project Structure

ABSA-BERT-Streamlit-Dashboard/
│
├── app.py
├── data/
├── output/
├── experiments.py/
├── requirements.txt
├── README.md
└── .gitignore


---

## How to Run the Project

### Step 1: Install dependencies

pip install -r requirements.txt


### Step 2: Run the application

streamlit run app.py

## Input Format
For batch processing, upload a CSV file containing a column named:

review

## Model Information
The sentiment classification model is based on a fine-tuned DistilBERT architecture.

Due to file size limitations, the trained model is not included in this repository. To run the project fully:

1. Download the trained model separately
2. Place it inside a folder named:

absa_model

## Use Cases
- Product review analysis
- Customer feedback monitoring
- Feature-level sentiment insights
- Decision support for product improvement

## Status
This project is fully functional and meets the required implementation goals for Aspect-Based Sentiment Analysis. Further improvements can include enhanced opinion extraction and more advanced visualizations.
