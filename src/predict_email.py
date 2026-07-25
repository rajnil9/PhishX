"""
PhishX Email Phishing Prediction Engine.
Processes email header/body features and outputs calibrated classification scores.
"""

import argparse
# pyrefly: ignore [missing-import]
import joblib
import json
import os
import re
import pandas as pd
from email_features import extract_email_metadata

# Try to import url features for stacked URL model verification
try:
    from url_features import extract_features as extract_url_features
except ImportError:
    extract_url_features = None

def generate_analysis_summary(prediction, features_df):
    headlines = {
        "ham": "Classified as Ham (Safe)",
        "promotional": "Classified as Promotional (Safe)",
        "phishing": "High Risk: Phishing Alert",
        "scam": "Critical Risk: Scam/Fraud Alert",
        "spam": "Warning: Spam Detected"
    }
    explanations = {
        "ham": "The email exhibits characteristics of normal conversational or transactional communication.",
        "promotional": "The email contains standard marketing or newsletter signatures without malicious intent.",
        "phishing": "The email contains urgent social engineering language commonly used to steal credentials.",
        "scam": "The email exhibits patterns associated with advance-fee fraud, extortion, or malicious payloads.",
        "spam": "The email contains bulk commercial or unsolicited content with low risk."
    }
    
    headline = headlines.get(prediction, "Unknown Classification")
    explanation = explanations.get(prediction, "No explanation available.")
    
    factors = []
    
    # Safely extract from DataFrame
    url_count = features_df['url_count'].iloc[0] if 'url_count' in features_df else 0
    uppercase_ratio = features_df['uppercase_ratio'].iloc[0] if 'uppercase_ratio' in features_df else 0
    exclamation_count = features_df['exclamation_count'].iloc[0] if 'exclamation_count' in features_df else 0
    dollar_count = features_df['dollar_count'].iloc[0] if 'dollar_count' in features_df else 0
    
    if url_count > 0:
        factors.append(f"Contains {url_count} embedded links or URLs.")
    if uppercase_ratio > 0.1:
        factors.append(f"Abnormal volume of capitalized letters (Ratio: {uppercase_ratio:.2f}).")
    if exclamation_count > 3:
        factors.append("High usage of exclamation marks indicating urgency or pressure.")
    if dollar_count > 0:
        factors.append("References financial transactions, payments, or currency.")
        
    urgent_words_count = sum([features_df[col].iloc[0] for col in features_df.columns if col.startswith('count_') and features_df[col].iloc[0] > 0])
    if urgent_words_count > 0:
        factors.append(f"Detected {urgent_words_count} high-risk social engineering or urgent keywords.")
        
    if not factors:
        if prediction in ["ham", "promotional"]:
            factors = [
                "Standard text length and vocabulary",
                "No suspicious embedded links or threat markers detected",
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
    parser = argparse.ArgumentParser(description="Predict if an email is spam or legitimate (ham).")
    parser.add_argument("text", type=str, help="The email text/content to check")
    args = parser.parse_args()
    
    # Define the absolute path to where the trained NLP pipeline is saved.
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'email_spam_model.pkl')
    url_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'phishshield_url_model.pkl')
    
    # Check if the model file actually exists before trying to load it.
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"Model not found at {model_path}. Please run train_email.py first."}))
        return
        
    # Load the pre-trained Pipeline (which includes the ColumnTransformer and RandomForestClassifier)
    pipeline = joblib.load(model_path)
    
    # Extract multi-modal metadata from the single input string
    input_series = pd.Series([args.text])
    X_input = extract_email_metadata(input_series)
    
    # Make initial prediction
    prediction = str(pipeline.predict(X_input)[0])
    
    # Get the confidence percentages
    probabilities = pipeline.predict_proba(X_input)[0]
    
    # Map the probability scores to their respective class names (ham, phishing, spam)
    class_probabilities = {
        str(cls): float(round(prob * 100, 2))
        for cls, prob in zip(pipeline.classes_, probabilities)
    }
    
    # =========================================================
    # STACKED SECURITY LAYER 1: CROSS-MODEL URL SCANNING
    # =========================================================
    # If the email contains a URL, we extract it and run it through our high-accuracy URL classifier.
    # If our URL model flags it as malicious with high confidence, we override the email's prediction.
    url_override = False
    url_override_reason = ""
    
    if extract_url_features and os.path.exists(url_model_path):
        # Extract URLs from the email body
        url_pattern = re.compile(r'https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:\/\S*)?')
        found_urls = url_pattern.findall(args.text)
        
        if found_urls:
            try:
                # Load the pre-trained URL model
                url_model = joblib.load(url_model_path)
                
                for url in found_urls:
                    # Clean trailing punctuation from the extracted URL (e.g. trailing periods or parentheses)
                    url_cleaned = url.rstrip(".,?!'\"()[]")
                    url_feats = extract_url_features(url_cleaned)
                    X_url = pd.DataFrame([url_feats])
                    url_pred = str(url_model.predict(X_url)[0])
                    url_probs = url_model.predict_proba(X_url)[0]
                    url_class_probs = {str(cls): prob for cls, prob in zip(url_model.classes_, url_probs)}
                    
                    # If the URL is predicted as malicious (phishing, malware, defacement) with confidence >= 35%, 
                    # we trigger the override. defacement/malware links in email are critical threat vectors.
                    conf = url_class_probs.get(url_pred, 0.0)
                    
                    if url_pred in ['phishing', 'malware', 'defacement'] and conf >= 0.35:
                        url_override = True
                        url_override_reason = f"Malicious link detected: '{url}' (URL model prediction: {url_pred}, confidence: {conf*100:.1f}%)"
                        prediction = "phishing"
                        break
            except Exception:
                pass # Fallback to standard ML if something goes wrong with the URL model
                
    # =========================================================
    # STACKED SECURITY LAYER 2: SHORT-TEXT THREAT SAFEGUARD
    # =========================================================
    # Short social engineering templates (phishing or scam) often try to dodge ML by being highly direct.
    # If the email is relatively short (< 500 characters) and has strong security threat signals, 
    # we enforce a safety override.
    if not url_override and len(args.text.strip()) < 500:
        text_lower = args.text.lower()
        
        # Phishing Signals (Urgent actions to steal credentials)
        phish_signals = ['urgent', 'compromised', 'verify', 'password', 'reset', 'suspended', 'locked', 'unauthorized', 'immediately', 'billing', 'security alert']
        matched_phish = [sig for sig in phish_signals if sig in text_lower]
        
        # Scam/Spam Signals (Fraudulent solicitations, Nigerian Prince, lottery fraud)
        spam_signals = ['million', 'transfer', 'winnings', 'beneficiary', 'inheritance', 'sum of', 'claim your', 'lottery', 'fund transfer', 'foster account']
        matched_spam = [sig for sig in spam_signals if sig in text_lower]
        
        # Enforce overrides based on signal matching
        if len(matched_phish) >= 2:
            prediction = "phishing"
            url_override = True
            url_override_reason = f"Short urgent notification containing phishing threat signals: {matched_phish}"
        elif len(matched_spam) >= 2:
            prediction = "spam"
            url_override = True
            url_override_reason = f"Short fraudulent solicitation containing spam signals: {matched_spam}"
            
    # =========================================================
    # STACKED SECURITY LAYER 3: LEGITIMATE SAFEGUARD & DE-BIASED OVERLAY (DISABLED)
    # =========================================================
    # This layer was historically needed for the 5-class model, but our new balanced 3-class 
    # model (trained on 142K rows) handles conversational text naturally.
    # Furthermore, checking for raw punctuation ('@', 'www.') fails on pre-processed NLP text,
    # causing aggressive false-negative overrides to 'ham'. This is now disabled.
    # if not url_override and prediction in ['phishing', 'spam']:
    #     pass


    # Format class probabilities if overridden
    if url_override:
        class_probabilities = {cls: 0.0 for cls in class_probabilities}
        class_probabilities[prediction] = 100.0
        
    # Generate the human-readable analysis summary
    analysis_summary = generate_analysis_summary(prediction, X_input)
        
    # Construct the final JSON response dictionary
    response = {
        "status": prediction,
        "class_probabilities": class_probabilities,
        "analysis_summary": analysis_summary
    }
    
    # Optionally append security reasons under an internal audit key for visibility (in CLI/Audit mode)
    if url_override_reason:
        response["security_alert"] = url_override_reason
        
    # Print the response to standard output as a beautifully formatted JSON string.
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
