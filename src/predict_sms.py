"""
PhishX SMS Spam & Phishing Prediction Engine.
Extracts metadata features and predicts probabilities using trained model artifact.
"""

import argparse
# pyrefly: ignore [missing-import]
import joblib
import json
import os
import sys
import pandas as pd
from sms_features import extract_sms_metadata

def generate_analysis_summary(prediction, features_df):
    headlines = {
        "ham": "Classified as Ham (Safe)",
        "spam": "Warning: Spam Detected",
        "smishing": "High Risk: Smishing Alert"
    }
    explanations = {
        "ham": "The SMS exhibits characteristics of normal conversational or transactional communication.",
        "spam": "The SMS contains unsolicited marketing, advertising, or promotional content.",
        "smishing": "The SMS contains deceptive social engineering language designed to harvest credentials or install malware."
    }
    
    headline = headlines.get(prediction, "Unknown Classification")
    explanation = explanations.get(prediction, "No explanation available.")
    
    factors = []
    
    # Safely extract from DataFrame
    url_count = features_df['url_count'].iloc[0] if 'url_count' in features_df else 0
    phone_count = features_df['phone_count'].iloc[0] if 'phone_count' in features_df else 0
    currency_count = features_df['currency_count'].iloc[0] if 'currency_count' in features_df else 0
    uppercase_ratio = features_df['uppercase_ratio'].iloc[0] if 'uppercase_ratio' in features_df else 0
    exclamation_count = features_df['exclamation_count'].iloc[0] if 'exclamation_count' in features_df else 0
    
    if url_count > 0:
        factors.append(f"Contains {url_count} embedded links or shortened URLs.")
    if phone_count > 0:
        factors.append("Contains phone numbers or shortcodes (Call-to-Action).")
    if currency_count > 0:
        factors.append("References financial transactions, lottery, or currency.")
    if uppercase_ratio > 0.1:
        factors.append(f"Abnormal volume of capitalized letters (Ratio: {uppercase_ratio:.2f}).")
    if exclamation_count > 2:
        factors.append("High usage of exclamation marks indicating urgency.")
        
    if not factors:
        if prediction == "ham":
            factors = [
                "Standard short-text vocabulary and length",
                "No suspicious embedded links or shortcodes detected",
                "Absence of urgency or financial extortion triggers"
            ]
        else:
            factors.append("General textual patterns match the detected class.")
            
    return {
        "headline": headline,
        "explanation": explanation,
        "key_factors": factors[:3]
    }

def main():
    # Configure stdout to use UTF-8 to prevent encoding crashes on Windows
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass # older python versions

    parser = argparse.ArgumentParser(description="Predict if an SMS is legitimate (ham), spam, or smishing.")
    parser.add_argument("text", type=str, help="The SMS message content to check")
    args = parser.parse_args()
    
    # Path to where the trained NLP pipeline is saved
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sms_spam_model.pkl')
    
    # Verify model presence
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"SMS model not found at {model_path}. Please run train_sms.py first."}))
        return
        
    # Load pipeline
    pipeline = joblib.load(model_path)
    
    # Extract metadata from input text
    input_series = pd.Series([args.text])
    X_input = extract_sms_metadata(input_series)
    
    # Predict label
    prediction = str(pipeline.predict(X_input)[0])
    
    # Predict probabilities
    probabilities = pipeline.predict_proba(X_input)[0]
    
    # Map probability scores to respective class names (ham, spam, smishing)
    class_probabilities = {
        str(cls): float(round(prob * 100, 2))
        for cls, prob in zip(pipeline.classes_, probabilities)
    }
    
    class_probabilities_raw = {
        str(cls): float(round(prob, 4))
        for cls, prob in zip(pipeline.classes_, probabilities)
    }
    
    step_3_str = ", ".join([f"{k}={v}%" for k, v in class_probabilities.items()])
    
    mathematical_breakdown = {
        "formula": "Calibrated Sigmoid / Decision Margin Scaling: P(k) = 1 / (1 + e^(A * f(x) + B))",
        "class_probabilities_raw": class_probabilities_raw,
        "step_by_step": [
            "Step 1: Evaluated high-dimensional sparse TF-IDF text matrix and mobile metadata using LinearSVC optimal decision hyperplane.",
            "Step 2: Applied Platt Sigmoid Scaling (CalibratedClassifierCV) to map raw decision function margins to true probabilities.",
            f"Step 3: Normalized class confidence scores: {step_3_str}."
        ]
    }
    
    # Generate the human-readable analysis summary
    analysis_summary = generate_analysis_summary(prediction, X_input)
    
    # Structure JSON response
    response = {
        "status": prediction,
        "class_probabilities": class_probabilities,
        "analysis_summary": analysis_summary,
        "mathematical_breakdown": mathematical_breakdown
    }
    
    # Print clean formatted JSON to standard output
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
