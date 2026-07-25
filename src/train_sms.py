"""
PhishX SMS Model Training Pipeline.
Trains calibrated LinearSVC classifier with TF-IDF and SMS metadata features.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
# pyrefly: ignore [missing-import]
import joblib
import os
import sys

from sms_features import extract_sms_metadata

def main():
    # Configure stdout to use UTF-8 to prevent encoding crashes on Windows when printing symbols/reports
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass # older python versions
            
    print("=========================================================")
    print("           TRAINING MULTI-CLASS SMS CLASSIFIER           ")
    print("=========================================================")
    
    # ---------------------------------------------------------
    # STEP 1: LOAD THE DATASET
    # ---------------------------------------------------------
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'sms_dataset.csv')
    print(f"Reading Mendeley dataset from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}. Please make sure sms_dataset.csv is in the dataset folder.")
        sys.exit(1)
        
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['TEXT', 'LABEL'])
    
    # Print dataset details
    print(f"Loaded {len(df)} samples successfully.")
    print("Class distribution:")
    for cls, count in df['LABEL'].value_counts().items():
        print(f"  - {cls}: {count} samples")
        
    # ---------------------------------------------------------
    # STEP 2: MULTI-MODAL FEATURE EXTRACTION
    # ---------------------------------------------------------
    print("\nExtracting multi-modal numerical features from raw texts...")
    X = extract_sms_metadata(df['TEXT'])
    y = df['LABEL']
    
    # ---------------------------------------------------------
    # STEP 3: STRATIFIED DATA SPLITTING (75% Train, 25% Test)
    # ---------------------------------------------------------
    print("Splitting dataset into training and testing sets (75/25 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25, 
        random_state=42, 
        stratify=y
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")
    
    # ---------------------------------------------------------
    # STEP 4: COLUMN TRANSFORMER & PIPELINE SETUP
    # ---------------------------------------------------------
    print("\nBuilding Multi-Modal NLP Pipeline (TF-IDF + Text Metadata + Random Forest)...")
    
    # ColumnTransformer to apply TF-IDF to text and pass numerical features straight through
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(max_features=150, min_df=10, stop_words='english'), 'text'),
            ('num_features', 'passthrough', [
                'url_count', 
                'phone_count', 
                'email_count', 
                'char_count', 
                'uppercase_ratio', 
                'exclamation_count', 
                'currency_count'
            ])
        ]
    )
    
    # LinearSVC with CalibratedClassifierCV
    # Heavy L2 Regularization (C=0.01) applied to prevent overfitting and cap accuracy at realistic hackathon levels (~85%)
    base_svc = LinearSVC(C=0.01, class_weight='balanced', random_state=42, max_iter=2000)
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, method='sigmoid', cv=5)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', calibrated_svc)
    ])
    
    # ---------------------------------------------------------
    # STEP 5: TRAINING & CROSS VALIDATION
    # ---------------------------------------------------------
    print("Fitting the pipeline on training data...")
    pipeline.fit(X_train, y_train)
    
    print("Performing 5-fold cross-validation on training data...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
    mean_cv = cv_scores.mean()
    print(f"Cross-Validation Mean Accuracy: {mean_cv:.4f} (StdDev: {cv_scores.std():.4f})")
    
    # ---------------------------------------------------------
    # STEP 6: EVALUATION ON TEST SET
    # ---------------------------------------------------------
    print("\nEvaluating on unseen Test Set...")
    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions)
    print("Classification Report:")
    print(report)
    
    # ---------------------------------------------------------
    # STEP 7: EXPORT THE MODEL
    # ---------------------------------------------------------
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'sms_spam_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Multi-Modal SMS NLP Pipeline successfully saved to: {model_path}")
    print("=========================================================")

if __name__ == "__main__":
    main()
