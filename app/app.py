import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from src.inference import NewsClassifier

# Page Config
st.set_page_config(
    page_title="AI News Classifier",
    page_icon="📰",
    layout="centered"
)

# Initialize the classifier (Cached so it only loads models once)
@st.cache_resource
def load_classifier():
    return NewsClassifier()

classifier = load_classifier()

# --- UI Header ---
st.title("📰 News Document Classifier")
st.markdown("""
Predict the category of news articles instantly using **Machine Learning** or **Deep Learning**. 
This system classifies news into **World**, **Sports**, **Business**, or **Sci/Tech**.
""")

st.divider()

# --- Sidebar ---
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model Architecture:",
    ("Machine Learning (SVM)", "Deep Learning (Neural Network)")
)

st.sidebar.info("""
**Tip:** The ML model is faster for short headlines, while the DL model often handles nuanced context better.
""")

# --- Main Input Area ---
input_text = st.text_area(
    "Enter news headline or snippet:",
    height=150,
    placeholder="Example: The stock market saw a record high today as tech companies reported strong earnings..."
)

# --- Prediction Logic ---
if st.button("Classify Article", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            m_type = 'ml' if "Machine Learning" in model_choice else 'dl'
            result = classifier.predict(input_text, model_type=m_type)
            
            st.markdown("### Prediction Result")
            st.success(f"**{result}**")

st.divider()

# --- Footer ---
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        Developed by <a href="https://github.com/Furqan09Ahmed/news-classification-system" target="_blank" style="color: #4A90E2; text-decoration: none; font-weight: bold;">
        Furqan Ahmed</a> | End-to-End NLP System
    </div>
    """,
    unsafe_allow_html=True
)