"""
PhishX SMS Feature Extraction Module.
Extracts structural, lexical, and urgency indicators from raw SMS text.
"""

import pandas as pd
import re

def extract_sms_metadata(text_series):
    """
    Extracts multi-modal numerical metadata and features from a pandas Series of raw SMS texts.
    Returns a DataFrame containing the raw text and the extracted features.
    This feature extractor is designed to work identically in training and in prediction pipelines,
    preventing any training-serving skew.
    """
    df = pd.DataFrame()
    df['text'] = text_series.astype(str)
    
    # 1. URL / Link Count
    # Matches http://, https://, www., or standard domain structures like link.co/abc
    url_pattern = re.compile(r'https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:\/\S*)?')
    df['url_count'] = df['text'].apply(lambda x: len(url_pattern.findall(x)))
    
    # 2. Phone Number / Shortcode Count
    # Matches numeric sequences between 5 and 15 digits (which covers shortcodes and full mobile numbers with optional country codes)
    phone_pattern = re.compile(r'\+?\b\d{5,15}\b|\+?\d{1,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}')
    df['phone_count'] = df['text'].apply(lambda x: len(phone_pattern.findall(x)))
    
    # 3. Email Address Count
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    df['email_count'] = df['text'].apply(lambda x: len(email_pattern.findall(x)))
    
    # 4. Length Metrics
    df['char_count'] = df['text'].str.len()
    
    # 5. Uppercase Ratio (often high in urgent spam messages shouting at the user)
    def calc_upper_ratio(text):
        text_str = str(text)
        if len(text_str) == 0:
            return 0.0
        # Calculate ratio based on alphabetic characters to avoid length distortion from numbers
        alpha_chars = [c for c in text_str if c.isalpha()]
        if not alpha_chars:
            return 0.0
        return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        
    df['uppercase_ratio'] = df['text'].apply(calc_upper_ratio)
    
    # 6. Special Characters
    df['exclamation_count'] = df['text'].str.count('!')
    
    # 7. Currency Symbols / Urgency Counts (matches $, £, €, or Indian Rupees symbols/strings)
    currency_pattern = re.compile(r'[\$£€]|Rs\.?|rupees?', re.IGNORECASE)
    df['currency_count'] = df['text'].apply(lambda x: len(currency_pattern.findall(x)))
    
    return df
