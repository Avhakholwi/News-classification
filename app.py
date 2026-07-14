import pandas as pd
import numpy as np
import re
import string
import joblib
import os
import streamlit as st

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
        font-weight: bold;
    }
    .model-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<div class="main-header">📰 News Category Classifier</div>', unsafe_allow_html=True)
st.markdown("""
    <p style="text-align: center; font-size: 1.1rem; color: #666; margin-bottom: 2rem;">
        Classify news articles using XGBoost, Random Forest, and Logistic Regression
    </p>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Select Model")
    model_choice = st.selectbox(
        "Choose a classifier:",
        ["XGBoost", "Random Forest", "Logistic Regression", "Ensemble (All Models)"],
        index=0
    )

    st.subheader("Input Method")
    input_method = st.radio(
        "How would you like to input text?",
        ["Type/Paste", "Upload CSV", "Sample Articles"],
        index=0
    )

    st.subheader("Confidence Threshold")
    confidence_threshold = st.slider(
        "Minimum confidence:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Predictions below this confidence level will be marked as 'Low Confidence'"
    )

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

# ============================================================================
# FUNCTIONS
# ============================================================================

def preprocess_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {
        'trained': False,
        'vectorizer': None,
        'xgboost': None,
        'random_forest': None,
        'logistic': None,
        'label_encoder': None,
        'categories': []
    }

    # Check if all model files exist
    model_files = {
        'vectorizer': 'tfidf_vectorizer.pkl',
        'xgboost': 'xgboost_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'logistic': 'logistic_regression_model.pkl',
        'label_encoder': 'label_encoder.pkl'
    }

    missing_files = []
    for name, filename in model_files.items():
        if not os.path.exists(filename):
            missing_files.append(filename)

    if missing_files:
        st.warning(f"Missing model files: {', '.join(missing_files)}")
        st.info("Please ensure all model files are in the same directory as this app.")
        return models

    try:
        # Load all models
        models['vectorizer'] = joblib.load('tfidf_vectorizer.pkl')
        models['xgboost'] = joblib.load('xgboost_model.pkl')
        models['random_forest'] = joblib.load('random_forest_model.pkl')
        models['logistic'] = joblib.load('logistic_regression_model.pkl')
        models['label_encoder'] = joblib.load('label_encoder.pkl')

        # Get categories from label encoder
        models['categories'] = models['label_encoder'].classes_.tolist()
        models['trained'] = True

        st.success(f"✅ Loaded {len(models['categories'])} categories successfully!")
        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return models

def predict_article(text, model_choice, models, confidence_threshold):
    """Predict category for a single article"""
    if not models.get('trained', False):
        return None

    # Preprocess and transform
    cleaned_text = preprocess_text(text)
    features = models['vectorizer'].transform([cleaned_text])

    # Get predictions based on model choice
    if model_choice == "XGBoost":
        predictions = models['xgboost'].predict_proba(features)[0]
        model_name = 'XGBoost'
    elif model_choice == "Random Forest":
        predictions = models['random_forest'].predict_proba(features)[0]
        model_name = 'Random Forest'
    elif model_choice == "Logistic Regression":
        predictions = models['logistic'].predict_proba(features)[0]
        model_name = 'Logistic Regression'
    else:  # Ensemble
        all_predictions = []
        for model in [models['xgboost'], models['random_forest'], models['logistic']]:
            if hasattr(model, 'predict_proba'):
                all_predictions.append(model.predict_proba(features)[0])

        predictions = np.mean(all_predictions, axis=0)
        model_name = 'Ensemble'

    categories = models['categories']

    # Get top predictions
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_categories = [categories[i] for i in top_indices]
    top_confidences = [predictions[i] for i in top_indices]

    best_confidence = top_confidences[0]
    if best_confidence < confidence_threshold:
        best_category = "⚠️ Low Confidence"
    else:
        best_category = top_categories[0]

    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'predicted_category': best_category,
        'confidence': best_confidence,
        'all_predictions': list(zip(top_categories, top_confidences)),
        'model_used': model_name
    }

def display_results(result):
    """Display prediction results"""
    if not result:
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🎯 Predicted Category")

        if result['confidence'] >= 0.7:
            color = "#28a745"
            emoji = "✅"
        elif result['confidence'] >= 0.5:
            color = "#ffc107"
            emoji = "⚠️"
        else:
            color = "#dc3545"
            emoji = "❌"

        st.markdown(f"""
            <div class="prediction-box" style="background-color: {color}20; border: 2px solid {color};">
                {emoji} {result['predicted_category']}
            </div>
        """, unsafe_allow_html=True)

        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.caption(f"Model: {result['model_used']}")

    with col2:
        st.markdown("### 📊 Top Predictions")

        df_predictions = pd.DataFrame(
            result['all_predictions'],
            columns=['Category', 'Confidence']
        )
        df_predictions['Confidence'] = df_predictions['Confidence'].apply(lambda x: f"{x:.1%}")

        st.dataframe(
            df_predictions,
            hide_index=True,
            use_container_width=True
        )

        # Confidence bar chart
        fig, ax = plt.subplots(figsize=(8, 2.5))
        categories = [p[0] for p in result['all_predictions']]
        confidences = [p[1] for p in result['all_predictions']]

        bars = ax.barh(categories, confidences, color='#1f77b4')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Confidence')
        ax.axvline(x=confidence_threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
        st.pyplot(fig)

    with st.expander("📝 View Text"):
        st.markdown("**Original Text:**")
        st.text_area("", result['text'], height=100, key="orig_text")
        st.markdown("**Cleaned Text:**")
        st.text_area("", result['cleaned_text'], height=80, key="clean_text")

# ============================================================================
# LOAD MODELS
# ============================================================================
models = load_models()

# Show model status in sidebar
with st.sidebar:
    if models.get('trained', False):
        st.success(f"✅ {len(models['categories'])} categories loaded")
        st.caption(f"Models: XGBoost, Random Forest, Logistic Regression")
    else:
        st.error("❌ Models not loaded")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if input_method == "Type/Paste":
    st.subheader("📝 Enter Article Text")
    user_input = st.text_area(
        "Paste or type your article text here:",
        height=200,
        placeholder="Enter news article text to classify..."
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        classify_button = st.button("🔍 Classify Article", use_container_width=True)

    if classify_button and user_input:
        if models.get('trained', False):
            with st.spinner("Classifying..."):
                result = predict_article(user_input, model_choice, models, confidence_threshold)
                display_results(result)
        else:
            st.error("Models not loaded. Please check that all model files exist.")
    elif classify_button:
        st.warning("Please enter some text to classify.")

elif input_method == "Upload CSV":
    st.subheader("📂 Upload CSV File")
    st.markdown("""
        Upload a CSV file with a `text` column containing news articles.
        The classifier will predict categories for each article.
    """)

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
                                    results.append({
                                        'text': text[:200] + '...' if len(text) > 200 else text,
                                        'category': result['predicted_category'],
                                        'confidence': f"{result['confidence']:.1%}"
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

else:  # Sample Articles
    st.subheader("📰 Sample Articles")
    st.markdown("Click a button below to classify a sample article:")

    sample_articles = {
        "Technology": "Google and Apple announce new partnership to develop AI-powered health monitoring features for smartphones. The initiative aims to detect early signs of heart disease and diabetes using sensor data.",

        "Politics": "The president signed a new executive order today that changes federal regulations for clean water. Environmental groups have pledged to challenge the order in court while industry leaders praised the move.",

        "Sports": "The Lakers secured a decisive victory in last night's game with LeBron James scoring 45 points. The win puts them in first place in the Western Conference with a 12-3 record.",

        "Entertainment": "Netflix has announced the release date for the highly anticipated third season of its hit series. The show's creator revealed that new characters will be introduced in the upcoming episodes.",

        "Health": "Researchers have made a breakthrough in cancer treatment developing a new therapy that targets tumor cells without damaging healthy tissue. Clinical trials are scheduled to begin next year.",

        "Business": "Tech stocks led the market rally today as the Federal Reserve signaled it would maintain its current policy stance. Analysts expect continued growth in the sector through the end of the year.",

        "World News": "The United Nations Security Council has called for an emergency meeting following the escalating conflict between neighboring nations. Diplomatic efforts are underway to prevent further escalation.",

        "Environment": "Climate scientists warn that global temperatures are on track to rise by 3 degrees Celsius by the end of the century if current emission trends continue. Urgent action is needed to meet Paris Agreement targets."
    }

    # Create columns for sample buttons
    cols = st.columns(4)
    for idx, (category, text) in enumerate(sample_articles.items()):
        col_idx = idx % 4
        with cols[col_idx]:
            if st.button(f"{category}", key=f"sample_{idx}"):
                if models.get('trained', False):
                    with st.spinner("Classifying..."):
                        result = predict_article(text, model_choice, models, confidence_threshold)
                        display_results(result)
                else:
                    st.error("Models not loaded.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("📰 News Category Classifier | Powered by XGBoost, Random Forest & Logistic Regression")

# ============================================================================
# SIDEBAR FOOTER
# ============================================================================
with st.sidebar:
    st.markdown("---")
    with st.expander("📊 Model Details"):
        if models.get('trained', False):
            st.markdown(f"**Categories:** {len(models['categories'])}")
            st.markdown("**Models loaded:**")
            st.markdown("- XGBoost ✅")
            st.markdown("- Random Forest ✅")
            st.markdown("- Logistic Regression ✅")
            st.markdown("- TF-IDF Vectorizer ✅")
            st.markdown("- Label Encoder ✅")
        else:
            st.info("Models not loaded. Check file paths.")

# ============================================================================
# ADDITIONAL INFO
# ============================================================================
if models.get('trained', False):
    with st.expander("ℹ️ Available Categories"):
        categories = models['categories']
        cols = st.columns(4)
        for idx, cat in enumerate(categories):
            col_idx = idx % 4
            with cols[col_idx]:
                st.markdown(f"- {cat}")
