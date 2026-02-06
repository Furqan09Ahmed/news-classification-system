import os
import re
import html
import joblib
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# AG News Mapping
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

class NewsClassifier:
    def __init__(self, ml_model_name='svm_model.pkl', dl_model_name='nn_model.h5', tokenizer_name='tokenizer.pkl'):
        """
        Initializes the classifier by loading the saved models from the models directory.
        """
        # Get the absolute path to the models directory (relative to this script)
        base_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # 1. Load ML Pipeline (TF-IDF + Model combined)
        ml_path = os.path.join(base_path, ml_model_name)
        if os.path.exists(ml_path):
            self.ml_pipeline = joblib.load(ml_path)
            print(f"✅ Loaded ML Model: {ml_model_name}")
        else:
            print(f"❌ ML Model not found at {ml_path}")

        # 2. Load DL Model
        dl_path = os.path.join(base_path, dl_model_name)
        if os.path.exists(dl_path):
            self.dl_model = tf.keras.models.load_model(dl_path)
            print(f"✅ Loaded DL Model: {dl_model_name}")
        else:
            print(f"❌ DL Model not found at {dl_path}")

        # 3. Load Tokenizer for DL
        tok_path = os.path.join(base_path, tokenizer_name)
        if os.path.exists(tok_path):
            with open(tok_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Loaded Tokenizer: {tokenizer_name}")
        else:
            print(f"❌ Tokenizer not found at {tok_path}")

    def _clean_text(self, text):
        """
        Standardized cleaning pipeline (Mirroring Phase 1 & 2)
        """
        text = html.unescape(text) # Remove &quot;, &amp; etc.
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special chars/numbers
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
        return text

    def predict(self, text, model_type='ml'):
        """
        Predicts the category of a given news string.
        model_type: 'ml' for SVM/Logistic, 'dl' for Neural Network
        """
        if not text:
            return "No text provided", 0.0

        cleaned_text = self._clean_text(text)

        if model_type == 'ml':
            # Scikit-learn pipeline handles vectorization internally
            prediction = self.ml_pipeline.predict([cleaned_text])[0]
            # Get probability if model supports it (Logistic/SVM with probability=True)
            # Since LinearSVC doesn't support predict_proba by default, we'll return label
            return LABEL_MAP[prediction]

        elif model_type == 'dl':
            # Convert text to sequences and pad
            seq = self.tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(seq, maxlen=70, padding='post')
            
            # Predict
            preds = self.dl_model.predict(padded, verbose=0)
            idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))
            
            return f"{LABEL_MAP[idx]} ({confidence*100:.2f}%)"

if __name__ == "__main__":
    # Quick Test Execution
    print("\n--- Testing Inference Layer ---")
    classifier = NewsClassifier()
    
    sample_news = "The tech giant announced a new smartphone with satellite connectivity today."
    
    print(f"\nInput: {sample_news}")
    print(f"ML Prediction: {classifier.predict(sample_news, 'ml')}")
    print(f"DL Prediction: {classifier.predict(sample_news, 'dl')}")