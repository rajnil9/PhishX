"""
PhishX Email Feature Extraction Module.
Extracts email header, body, URL counts, and domain features.
"""

import pandas as pd
import re

def extract_email_metadata(text_series):
    """
    Extracts multi-modal numerical metadata and features from a pandas Series of raw email texts.
    Returns a DataFrame containing the raw text and the extracted features.
    """
    df = pd.DataFrame()
    df['text'] = text_series.astype(str)
    
    # 1. URL Extraction / Link count
    # Count occurrences of 'http://', 'https://', or 'www.'
    df['url_count'] = df['text'].str.count(r'http[s]?://|www\.')
    
    # 2. Length metrics
    df['char_count'] = df['text'].str.len()
    
    # 3. Uppercase ratio (shouting) in a vectorized way
    upper_counts = df['text'].str.count(r'[A-Z]')
    df['uppercase_ratio'] = upper_counts / (df['char_count'].replace(0, 1))
    
    # 4. Special Characters typically abused in spam
    df['exclamation_count'] = df['text'].str.count('!')
    df['dollar_count'] = df['text'].str.count(r'\$')
    
    # 5 & 6. Urgent/Threat & Financial/Scam keywords (Social Engineering signals)
    # Lowercase text only ONCE to avoid 31 redundant lowercasing passes
    lower_text = df['text'].str.lower()
    
    urgent_words = [
        'urgent', 'compromised', 'verify', 'secure', 'suspended', 
        'alert', 'password', 'reset', 'billing', 'unauthorized', 
        'locked', 'immediately', 'action', 'security', 'notification'
    ]
    for word in urgent_words:
        df[f'count_{word}'] = lower_text.str.count(rf'\b{word}\b')
        
    scam_words = [
        'bitcoin', 'wallet', 'invoice', 'payment', 'bank', 
        'account', 'transfer', 'dollars', 'fund', 'claim', 
        'winnings', 'money', 'inherit', 'lottery', 'winner', 'gift'
    ]
    for word in scam_words:
        df[f'count_{word}'] = lower_text.str.count(rf'\b{word}\b')
    
    return df

