import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_models():
    """Load XGBoost and Logistic Regression models"""
    models = {
        'trained': False,
        'vectorizer': None,
        'xgboost': None,
        'logistic': None,
        'label_encoder': None,
        'categories': []
    }

    # Required files
    model_files = {
        'vectorizer': 'tfidf_vectorizer.pkl',
        'xgboost': 'xgboost_model.pkl',
        'logistic': 'logistic_regression_model.pkl',
        'label_encoder': 'label_encoder.pkl'
    }

    missing_files = [f for f in model_files.values() if not os.path.exists(f)]
    if missing_files:
        st.warning(f"Missing model files: {', '.join(missing_files)}")
        st.info("Please ensure all model files are in the same directory as this app.")
        return models

    try:
        models['vectorizer'] = joblib.load(model_files['vectorizer'])
        models['xgboost'] = joblib.load(model_files['xgboost'])
        models['logistic'] = joblib.load(model_files['logistic'])
        models['label_encoder'] = joblib.load(model_files['label_encoder'])

        models['categories'] = models['label_encoder'].classes_.tolist()
        models['trained'] = True

        st.success(f"✅ Loaded {len(models['categories'])} categories successfully!")
        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return models
