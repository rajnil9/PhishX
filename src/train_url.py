"""
PhishX URL Phishing Model Training Pipeline.
Trains ensemble classifier using lexical and structural URL features.
"""

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
# pyrefly: ignore [missing-import]
import joblib
import os
from url_features import extract_features

def main():
    # ---------------------------------------------------------
    # STEP 1: LOAD THE DATASET
    # ---------------------------------------------------------
    # Resolve the path to the massive dataset CSV file.
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'url_dataset.csv')
    print(f"Reading dataset from {dataset_path}...")
    
    # Load the dataset into a Pandas DataFrame.
    df = pd.read_csv(dataset_path)
    
    # ---------------------------------------------------------
    # STEP 2: DATA BALANCING & SUBSAMPLING
    # ---------------------------------------------------------
    print("Class distribution before balancing:")
    class_counts = df['type'].value_counts()
    print(class_counts)
    
    min_class_count = class_counts.min()
    N = min(min_class_count, 25000)
    print(f"Sampling exactly {N} samples per class...")
    
    subsets = []
    # pyrefly: ignore [missing-import]
    import tldextract
    for t in df['type'].unique():
        class_data = df[df['type'] == t]
        
        # Data Augmentation to prevent length-shortcut bias on benign URLs
        if t == 'benign':
            augmented_urls = []
            for u in class_data['url'].sample(min(len(class_data), 25000), random_state=42):
                try:
                    ext = tldextract.extract(str(u))
                    if ext.domain and ext.suffix:
                        bare_domain = f"{ext.domain}.{ext.suffix}"
                        augmented_urls.append(bare_domain)
                except Exception:
                    pass
            augmented_urls = list(set(augmented_urls))
            aug_df = pd.DataFrame({'url': augmented_urls, 'type': ['benign'] * len(augmented_urls)})
            
            # Combine original and augmented benign, remove duplicates, and sample exactly N
            combined_benign = pd.concat([class_data, aug_df]).drop_duplicates(subset=['url'])
            sub = combined_benign.sample(N, random_state=42)
            subsets.append(sub)
        else:
            sub = class_data.sample(N, random_state=42)
            subsets.append(sub)
            
    df = pd.concat(subsets)
    print("Class distribution after balancing:")
    print(df['type'].value_counts())
    
    # ---------------------------------------------------------
    # STEP 3: FEATURE EXTRACTION
    # ---------------------------------------------------------
    print("Extracting lexical features...")
    # Apply our custom feature extraction function to every URL in the dataset.
    # This transforms the raw URL strings into numerical dictionaries (e.g., {'length': 25, 'slash_count': 3})
    feature_dicts = df['url'].apply(extract_features)
    
    # Convert the list of dictionaries into a 2D Pandas DataFrame (Features/X)
    X = pd.DataFrame(feature_dicts.tolist())
    # Grab the target labels (Y) and encode them to integers for LightGBM
    le = LabelEncoder()
    y = le.fit_transform(df['type'])
    
    # ---------------------------------------------------------
    # STEP 4: DATA SPLITTING & CROSS VALIDATION
    # ---------------------------------------------------------
    print("Splitting dataset into training and testing sets (75% train, 25% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print("Training LGBMClassifier with class balancing and tuned capacity...")
    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=8,
        min_child_samples=15,
        class_weight='balanced',
        objective='multiclass',
        random_state=42
    )
    
    # Perform 5-fold cross-validation on the training set
    print("Performing 5-fold cross-validation on training data...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")
    
    # Train (fit) the model on the training set
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # STEP 5: EVALUATION & METRICS
    # ---------------------------------------------------------
    print("\nClassification Report (Test Set):")
    # Ask the trained model to predict classifications for the unseen test set
    predictions = model.predict(X_test)
    
    # Decode integers back to strings for readable reports
    y_test_labels = le.inverse_transform(y_test)
    predictions_labels = le.inverse_transform(predictions)
    
    # Print out Precision, Recall, and F1-Scores for each class.
    print(classification_report(y_test_labels, predictions_labels))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_labels, predictions_labels))
    
    print("Feature Importances:")
    # Extract how much weight/importance the model assigned to each feature.
    # This helps us understand *why* the model makes its decisions (e.g., slash_count is highly predictive).
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))
    
    # ---------------------------------------------------------
    # STEP 6: EXPORT THE MODEL
    # ---------------------------------------------------------
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    # Ensure the models directory exists; if not, create it.
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the fully trained model and label encoder to disk as a bundled .pkl (Pickle) file.
    # This allows predict.py to load the exact same model instantly without retraining.
    model_path = os.path.join(model_dir, 'phishshield_url_model.pkl')
    artifact = {'model': model, 'label_encoder': le}
    joblib.dump(artifact, model_path)
    print(f"\nModel bundled and saved to {model_path}")

if __name__ == "__main__":
    main()
