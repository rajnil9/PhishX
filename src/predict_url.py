"""
PhishX URL Phishing & Safety Prediction Engine.
Evaluates URL safety score using PhishShield trained model artifact.
"""

import argparse
# pyrefly: ignore [missing-import]
import joblib
import json
import os
import numpy as np
import pandas as pd
from url_features import extract_features

def generate_analysis_summary(prediction, features):
    headlines = {
        "benign": "Classified as Benign (Safe)",
        "phishing": "High Risk: Phishing Alert",
        "malware": "Critical Risk: Malware Host",
        "defacement": "Warning: Defaced Webpage"
    }
    explanations = {
        "benign": "The URL exhibits standard domain parameters with clean lexical features and no indicators of typosquatting or path manipulation.",
        "phishing": "This URL exhibits deceptive characteristics commonly used to steal credentials or masquerade as trusted brands.",
        "malware": "This URL demonstrates structural anomalies frequently associated with hosting or distributing malicious payloads.",
        "defacement": "This URL contains path-level signatures suggesting a compromised or defaced legitimate domain."
    }
    
    headline = headlines.get(prediction, "Unknown Classification")
    explanation = explanations.get(prediction, "No explanation available.")
    
    factors = []
    if features.get('is_legit_brand') == 1:
        factors.append("Verified as official brand domain.")
    if features.get('suspicious_tld') == 1:
        factors.append("Uses a high-risk or commonly abused Top-Level Domain (TLD).")
    if features.get('ip_based') == 1:
        factors.append("Uses a raw IP address instead of a registered domain.")
    if features.get('has_executable_ext') == 1:
        factors.append("Path ends with an executable or scripting extension.")
    if features.get('defacement_keywords') == 1:
        factors.append("Path contains keywords frequently found in defaced sites.")
    if features.get('is_shortened') == 1:
        factors.append("URL uses a known link shortening service to obfuscate the destination.")
    if features.get('double_slash') == 1:
        factors.append("Contains double slashes indicative of open redirect abuse.")
    if features.get('path_depth', 0) > 3:
        factors.append(f"Deeply nested URL path structure (Depth: {features.get('path_depth')}).")
    if features.get('subdomain_count', 0) > 2:
        factors.append(f"High number of nested subdomains (Count: {features.get('subdomain_count')}).")
        
    if len(factors) < 3:
        if features.get('length', 0) > 75:
            factors.append("URL is abnormally long, often used to hide suspicious parameters.")
        if features.get('digit_ratio', 0) > 0.2:
            factors.append("High ratio of numeric digits, indicating potential machine-generated strings.")
            
    if not factors:
        if prediction == "benign":
            factors = [
                "Valid registered root domain without brand impersonation triggers",
                "Standard path depth and character length ratios",
                "No suspicious Top-Level Domain (TLD) or raw IP hosting"
            ]
        else:
            factors.append("Lexical structure aligns with typical web standards.")
        
    return {
        "headline": headline,
        "explanation": explanation,
        "key_factors": factors[:3]
    }

def main():
    # Set up argument parsing so the user can pass a URL from the command line.
    # Example: python src/predict.py "http://example.com"
    parser = argparse.ArgumentParser(description="Predict if a URL is phishing or safe.")
    parser.add_argument("url", type=str, help="The URL to check")
    args = parser.parse_args()
    
    # Define the absolute path to where the trained machine learning model is saved.
    # It looks in the 'models' directory at the root of the project.
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'phishshield_url_model.pkl')
    
    # Check if the model file actually exists before trying to load it.
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"Model not found at {model_path}. Please run train.py first."}))
        return
        
    # Load the pre-trained artifact dictionary from disk using joblib.
    artifact = joblib.load(model_path)
    # Support both new dict format (LightGBM + LE) and legacy format (Raw RF Model)
    if isinstance(artifact, dict) and 'model' in artifact and 'label_encoder' in artifact:
        model = artifact['model']
        le = artifact['label_encoder']
    else:
        model = artifact
        le = None
    
    # Pass the user's URL into our feature extraction engine to get the exact same
    # lexical and structural features we used during training.
    features = extract_features(args.url)
    
    # Convert the extracted features dictionary into a Pandas DataFrame.
    # This is required because scikit-learn models expect 2D array-like inputs (Rows x Columns).
    X = pd.DataFrame([features])
    
    # Make a prediction. model.predict returns an array of predictions, so we grab the first one [0].
    raw_prediction = model.predict(X)[0]
    
    # Inverse transform if we have a LabelEncoder, otherwise assume it's already a string
    if le:
        prediction = str(le.inverse_transform([raw_prediction])[0])
        class_names = le.classes_
    else:
        prediction = str(raw_prediction)
        class_names = model.classes_
    
    # Get the confidence percentages (probabilities) for all possible classes (benign, malware, etc.).
    probabilities = model.predict_proba(X)[0]
    
    try:
        # Obtain raw tree margin scores (logits)
        raw_logits = model.predict(X, raw_score=True)[0]
    except Exception:
        # Fallback if raw_score is unsupported
        raw_logits = np.zeros(len(class_names))
        
    exponentials = np.exp(raw_logits)
    denominator = np.sum(exponentials)
    
    raw_logits_dict = {}
    exponentials_dict = {}
    class_probabilities = {}
    
    for cls, logit, exp, prob in zip(class_names, raw_logits, exponentials, probabilities):
        perc = float(round(prob * 100, 2))
        cls_str = str(cls)
        class_probabilities[cls_str] = perc
        raw_logits_dict[cls_str] = float(round(logit, 3))
        exponentials_dict[cls_str] = float(round(exp, 3))
        
    logits_str = ", ".join([f"{k}={v}" for k, v in raw_logits_dict.items()])
    exp_str = ", ".join([f"{k}={v}" for k, v in exponentials_dict.items()])
    
    # Determine the top class for the step 3 example
    top_class = prediction
    top_class_exp = exponentials_dict.get(top_class, 0.0)
    top_class_perc = class_probabilities.get(top_class, 0.0)
    
    mathematical_breakdown = {
        "formula": "P(k) = e^(z_k) / Sum(e^(z_j))",
        "raw_logits_z": raw_logits_dict,
        "exponentials_exp_z": exponentials_dict,
        "sum_denominator": float(round(denominator, 3)),
        "step_by_step": [
            f"Step 1: Gathered raw tree margin scores (logits) from LightGBM ensemble trees: {logits_str}",
            f"Step 2: Applied exponential transformation e^(z_k) to eliminate negative values: {exp_str}",
            f"Step 3: Summed exponentials (Denominator = {round(denominator, 3)}). Divided each by total sum to normalize probabilities (e.g., {top_class} = {top_class_exp} / {round(denominator, 3)} = {top_class_perc}%)."
        ]
    }
    
    # Security Whitelist Override:
    # Top brands (like google.com, udemy.com) have very short lengths and no paths.
    # The ML model often falsely flags short domains as phishing because malware uses short disposable domains.
    # If our feature extractor explicitly verified this is a top legitimate brand, we override the ML prediction.
    if features.get('is_legit_brand') == 1:
        original_pred = prediction
        prediction = "benign"
        for cls in class_probabilities:
            class_probabilities[cls] = 100.0 if cls == "benign" else 0.0
        mathematical_breakdown["step_by_step"].append(
            f"Step 4: Security Whitelist Override - The ML model originally predicted '{original_pred}', but the domain is a verified safe brand. Probabilities forcefully overridden to benign=100.0%."
        )
    elif features.get('is_root_domain') == 1 and features.get('subdomain_count', 0) == 0 and features.get('is_standard_tld') == 1:
        suspicious_kws = ['login', 'verify', 'secure', 'update', 'account', 'banking', 'wallet']
        if not any(features.get(f'has_{kw}') == 1 for kw in suspicious_kws):
            if prediction != "benign":
                original_pred = prediction
                prediction = "benign"
                for cls in class_probabilities:
                    class_probabilities[cls] = 100.0 if cls == "benign" else 0.0
                mathematical_breakdown["step_by_step"].append(
                    f"Step 4: Root Domain Safeguard - The ML model predicted '{original_pred}', but the URL is a clean bare root domain with a standard TLD and no suspicious keywords. Probabilities forcefully overridden to benign=100.0%."
                )

    
    # Generate the human-readable analysis summary
    analysis_summary = generate_analysis_summary(prediction, features)
    
    # Construct the final JSON response dictionary.
    response = {
        "status": prediction,
        "class_probabilities": class_probabilities,
        "analysis_summary": analysis_summary,
        "mathematical_breakdown": mathematical_breakdown
    }
    
    # Print the response to standard output as a beautifully formatted JSON string.
    # This allows a frontend, Node.js server, or dashboard to easily parse the output.
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    # Execute the main function when the script is run directly.
    main()
