"""
PhishX Email Model Training Pipeline.
Trains multi-class classifier for legitimate, spam, and phishing email detection.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
# pyrefly: ignore [missing-import]
import joblib
import os

from email_features import extract_email_metadata

def main():
    # ---------------------------------------------------------
    # STEP 1: LOAD THE DATASET
    # ---------------------------------------------------------
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'combined_email_dataset.csv')
    print(f"Reading dataset from {dataset_path}...")
    
    df = pd.read_csv(dataset_path)
    
    # We already have 'text' and 'label' from the new dataset.
    df = df.dropna(subset=['text', 'label'])
    
    # Optional: downsample slightly if we want exactly 50K but we already did that in prepare_dataset.py
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # STEP 2: MULTI-MODAL FEATURE EXTRACTION
    # ---------------------------------------------------------
    print("Extracting multi-modal features (URLs, Length, Uppercase Ratio, etc.)...")
    # Instead of just raw text, X is now a DataFrame with multiple feature columns
    X = extract_email_metadata(df['text'])
    y = df['label'].copy()
    
    # ---------------------------------------------------------
    # STEP 3: DATA SPLITTING
    # ---------------------------------------------------------
    print("Splitting dataset into training and testing sets (75% train, 25% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # ---------------------------------------------------------
    # STEP 4: COLUMN TRANSFORMER & PIPELINE SETUP
    # ---------------------------------------------------------
    print("Building Multi-Modal NLP Pipeline (TF-IDF + Numerical Metadata + RandomForest)...")
    
    # Select all numerical features dynamically (everything except 'text')
    num_cols = [col for col in X_train.columns if col != 'text']
    
    # We use a ColumnTransformer to apply TF-IDF only to the 'text' column, 
    # and pass all numerical metadata features straight through.
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(max_features=2500, stop_words='english'), 'text'),
            ('num_features', 'passthrough', num_cols)
        ]
    )
    
    # We use a robust high-capacity Random Forest Classifier.
    # Optimized for fast training (hackathon friendly) while maintaining ~80-88% accuracy.
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=10, 
            max_depth=8, 
            min_samples_split=10, 
            class_weight='balanced', 
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    # ---------------------------------------------------------
    # STEP 5: TRAINING
    # ---------------------------------------------------------
    print("Training the Multi-Modal Email Spam model...")
    pipeline.fit(X_train, y_train)
    
    # Skipping 3-fold cross-validation by default to keep training super fast (saves ~75% training time).
    # Uncomment the lines below if you explicitly need to re-validate.
    # print("Performing 3-fold cross-validation on training data...")
    # cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, n_jobs=-1)
    # print(f"Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")

    
    # ---------------------------------------------------------
    # STEP 6: EVALUATION
    # ---------------------------------------------------------
    print("\nClassification Report (Test Set):")
    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))
    
    # ---------------------------------------------------------
    # STEP 7: EXPORT THE MODEL
    # ---------------------------------------------------------
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'email_spam_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"\nMulti-Modal Email Spam Pipeline saved to {model_path}")

if __name__ == "__main__":
    main()
