# 📰 News Category Classification using Machine Learning

## Overview

This project is a Machine Learning-based News Category Classification system that automatically classifies news articles into predefined categories based on their headlines and descriptions. The model uses Natural Language Processing (NLP) techniques to preprocess text, extract meaningful features, and accurately predict the category of unseen news articles.

The project demonstrates an end-to-end machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and deployment readiness.

---

## Features

- Text preprocessing using NLP techniques
- TF-IDF Vectorization
- Multiple Machine Learning models
- Model performance comparison
- Hyperparameter tuning
- Prediction on new news articles
- Model serialization using Joblib
- Deployment-ready architecture

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib
- Matplotlib
- Seaborn

---

## Dataset

The dataset contains news articles with their corresponding categories.

Typical columns include:

- Headline
- Short Description
- Category

---

## Project Workflow

### 1. Data Loading

- Import dataset
- Explore missing values
- Remove duplicates
- Data cleaning

### 2. Text Preprocessing

The text was cleaned using several Natural Language Processing techniques:

- Lowercasing
- Removing punctuation
- Removing numbers
- Tokenization
- Stopword removal
- Lemmatization

---

### 3. Feature Engineering

The cleaned text was converted into numerical features using:

- TF-IDF Vectorizer

---

### 4. Model Training

Several machine learning algorithms were trained and compared, including:

- Logistic Regression
- Multinomial Naïve Bayes
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost *(if included in your project)*

---

### 5. Model Evaluation

Models were evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

The best-performing model was selected and saved for deployment.

---

## Project Structure

```
News-Category-Classification/
│
├── dataset/
│   └── News_Category_Dataset.csv
│
├── models/
│   ├── news_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── News_Classification.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/news-category-classification.git
```

Navigate to the project folder

```bash
cd news-category-classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the application

```bash
python app.py
```

or if using Streamlit

```bash
streamlit run app.py
```

---

## Example Prediction

**Input**

```
Apple unveils its latest AI-powered iPhone with significant performance improvements.
```

**Output**

```
Category: Technology
```

---

## Future Improvements

- Deep Learning using LSTM
- BERT Transformer model
- RoBERTa implementation
- Real-time news classification
- Web application deployment
- API integration
- Cloud deployment

---

## Skills Demonstrated

- Data Cleaning
- Natural Language Processing (NLP)
- Feature Engineering
- Machine Learning
- Text Classification
- Model Evaluation
- Model Deployment
- Python Programming
- Data Visualization

---

## Author

**Avhakholwi Maladze**

Data Scientist | Machine Learning Engineer | Data Analyst

- LinkedIn: *Add your LinkedIn URL*
- GitHub: *Add your GitHub URL*

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Scikit-learn
- NLTK
- Pandas
- NumPy
- Open-source Machine Learning community

- https://news-classification-r9wayjbnhnhabsvxjyfctj.streamlit.app/
