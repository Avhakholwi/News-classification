import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(page_title="News Classifier", page_icon="📰")
st.title("📰 News Category Classifier")

# ==============================
# TEXT PREPROCESSING
# ==============================
def preprocess_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    models = {
        'trained': False,
        'vectorizer': None,
        'xgboost': None,
        'logistic': None,
        'label_encoder': None,
        'categories': []
    }

    model_files = {
        'vectorizer': 'tfidf_vectorizer.pkl',
        'xgboost': 'xgboost_model.pkl',
        'logistic': 'logistic_regression_model.pkl',
        'label_encoder': 'label_encoder.pkl'
    }

    missing_files = [f for f in model_files.values() if not os.path.exists(f)]
    if missing_files:
        st.warning(f"Missing model files: {', '.join(missing_files)}")
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

models = load_models()

# ==============================
# SIDEBAR CONFIG
# ==============================
with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.selectbox(
        "Choose a classifier:",
        ["XGBoost", "Logistic Regression", "Ensemble (Both Models)"]
    )
    confidence_threshold = st.slider(
        "Minimum confidence:",
        0.0, 1.0, 0.5, 0.05
    )
    input_method = st.radio(
        "Input Method:",
        ["Type/Paste", "Upload CSV"],
        index=0
    )

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_article(text, model_choice, models, confidence_threshold):
    if not models.get('trained', False):
        return None

    cleaned_text = preprocess_text(text)
    features = models['vectorizer'].transform([cleaned_text])

    if model_choice == "XGBoost":
        predictions = models['xgboost'].predict_proba(features)[0]
        model_name = "XGBoost"
    elif model_choice == "Logistic Regression":
        predictions = models['logistic'].predict_proba(features)[0]
        model_name = "Logistic Regression"
    else:  # Ensemble
        preds_xgb = models['xgboost'].predict_proba(features)[0]
        preds_log = models['logistic'].predict_proba(features)[0]
        predictions = np.mean([preds_xgb, preds_log], axis=0)
        model_name = "Ensemble"

    categories = models['categories']
    top_idx = np.argmax(predictions)
    best_category = categories[top_idx]
    best_confidence = predictions[top_idx]

    if best_confidence < confidence_threshold:
        best_category = "⚠️ Low Confidence"

    return best_category, best_confidence, model_name

# ==============================
# MAIN APP
# ==============================
if input_method == "Type/Paste":
    st.subheader("📝 Enter Article Text")
    user_input = st.text_area("Paste or type your article text here:", height=200)

    if st.button("🔍 Classify Article"):
        if models.get('trained', False) and user_input.strip():
            with st.spinner("Classifying..."):
                category, confidence, model_used = predict_article(
                    user_input, model_choice, models, confidence_threshold
                )
                st.success(f"Predicted Category ({model_used}): {category}")
                st.metric("Confidence", f"{confidence:.1%}")
        else:
            st.error("Models not loaded or no text provided.")

else:  # Upload CSV
    st.subheader("📂 Upload CSV File")
    st.markdown("Upload a CSV file with a `text` column containing news articles.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                st.success(f"Loaded {len(df)} articles.")
                st.dataframe(df.head())

                if st.button("🔍 Classify All Articles"):
                    if models.get('trained', False):
                        with st.spinner("Classifying articles..."):
                            results = []
                            for text in df['text']:
                                result = predict_article(text, model_choice, models, 0.0)
                                if result:
                                    category, confidence, model_used = result
                                    results.append({
                                        'text': text[:200] + '...' if len(text) > 200 else text,
                                        'category': category,
                                        'confidence': f"{confidence:.1%}",
                                        'model': model_used
                                    })

                            result_df = pd.DataFrame(results)

                        st.subheader("📊 Classification Results")
                        st.dataframe(result_df)

                        # Download button
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="classified_articles.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Models not loaded.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
