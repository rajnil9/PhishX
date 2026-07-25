# PhishX / PhishShield ML Security Suite

`PhishX` (formerly `PhishShield_ML`) is an enterprise-grade Machine Learning Security Suite designed to detect web and messaging threats across three critical channels: **URLs**, **Emails**, and **SMS (Smishing)** messages.

---

## 🚀 Key Features & Models

### 1. URL Threat Classifier (LightGBM)
* **Classes**: `benign`, `phishing`, `malware`, `defacement`.
* **Model**: High-performance `lightgbm.LGBMClassifier` bundled with `LabelEncoder`.
* **Feature Extraction**: 31 lexical, structural, typosquatting, and path-specific defacement features (`path_length`, `has_executable_ext`, `path_depth`, `defacement_keywords`, etc.).
* **Model Size**: Highly optimized (~2.8 MB).
* **Performance**: **84% Generalization Accuracy**, **86% Defacement F1-score**.

### 2. Multi-Modal Email Threat Engine (NLP + 3-Layer Safeguards)
* **Classes**: `ham`, `phishing`.
* **Model**: `Calibrated LinearSVC` trained on the `Phishing_Email.csv` dataset.
* **Architecture**: Multi-modal `ColumnTransformer` combining TF-IDF (5,000 features) and 30 numerical structural/density metrics.
* **3-Layer Security Pipeline**:
  - **Layer 1 (Stacked Link Scanning)**: Runs embedded links through the trained URL model.
  - **Layer 2 (Urgency & Threat Safeguard)**: Catches short high-risk social engineering templates.
  - **Layer 3 (False Positive Overlay)**: Automatically downgrades messages with no active call-to-action or threat indicators.

### 3. Mobile SMS / Smishing Classifier (NLP)
* **Classes**: `ham`, `spam`, `smishing`.
* **Model**: `Calibrated LinearSVC` trained on 10,191 modern mobile texts.
* **Feature Extraction**: TF-IDF combined with short-text mobile metrics (phone counts, shortcodes, URL presence, shouting ratio).
* **Performance**: **~93% Generalization Accuracy** (100% recall on safe `ham` messages).

---

## 🛠️ Quickstart & Setup

### Environment Activation
```powershell
# In PowerShell:
.\venv\Scripts\Activate.ps1

# Install requirements if needed:
python -m pip install -r requirements.txt
```

### Model Training
```powershell
# Train URL LightGBM Classifier
python src/train_url.py

# Train Email Multi-Modal Classifier
python src/train_email.py

# Train SMS Smishing Classifier
python src/train_sms.py
```

### Live Predictions
```powershell
# Test URL Prediction
python src/predict_url.py "http://google.com"

# Test Email Prediction
python src/predict_email.py "Urgent account suspension alert: please verify your credentials."

# Test SMS Prediction
python src/predict_sms.py "Claim your free $500 gift voucher now by calling +1-800-555-0199"
```

---

## 📖 Documentation
- [project_context.md](file:///c:/Users/rajni/OneDrive/Documents/VS/PhishX/project_context.md) — Comprehensive technical architecture, dataset details, and debugging history.
- [models_study_guide.md](file:///c:/Users/rajni/OneDrive/Documents/VS/PhishX/models_study_guide.md) — Detailed educational study guide & architectural decision matrix.

---

*PhishX Threat Detection System v1.0*
