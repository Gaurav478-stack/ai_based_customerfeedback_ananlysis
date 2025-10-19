import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.processed_data = None
    
    def load_amazon_data(self, filepath):
        """Load and preprocess Amazon review data"""
        df = pd.read_excel(filepath)
        print(f"Loaded data: {df.shape}")
        
        # Basic cleaning
        df_clean = self.clean_data(df)
        
        # Feature engineering
        df_processed = self.create_features(df_clean)
        
        self.processed_data = df_processed
        return df_processed
    
    def clean_data(self, df):
        """Clean the raw data"""
        df_clean = df.copy()
        
        # Handle price columns
        price_columns = ['discounted_price', 'actual_price']
        for col in price_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.replace('â‚¹', '').str.replace(',', '').str.strip()
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Clean ratings
        if 'rating' in df_clean.columns:
            df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
        
        # Clean text columns
        text_columns = ['product_name', 'about_product', 'review_title', 'review_content']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str)
        
        return df_clean
    
    def create_features(self, df):
        """Create features for training"""
        df_featured = df.copy()
        
        # Sentiment labels from ratings
        if 'rating' in df_featured.columns:
            df_featured['sentiment_label'] = df_featured['rating'].apply(
                lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral'
            )
        
        # Review credibility scores (simulated - in real scenario, this would be labeled data)
        if 'review_content' in df_featured.columns:
            df_featured['credibility_score'] = df_featured['review_content'].apply(
                self.estimate_credibility
            )
        
        # Combine review title and content
        if all(col in df_featured.columns for col in ['review_title', 'review_content']):
            df_featured['full_review'] = df_featured['review_title'] + ' ' + df_featured['review_content']
        elif 'review_content' in df_featured.columns:
            df_featured['full_review'] = df_featured['review_content']
        
        return df_featured
    
    def estimate_credibility(self, text):
        """Estimate credibility score based on text characteristics"""
        if not text or text == 'nan':
            return 0.5
        
        text = str(text)
        features = {
            'length': min(len(text) / 500, 1.0),  # Longer reviews more credible
            'has_details': 1.0 if any(word in text.lower() for word in ['because', 'since', 'therefore']) else 0.3,
            'has_specifics': 1.0 if re.search(r'\b(\d+|\d+\.\d+|[A-Z][a-z]+)\b', text) else 0.4,
            'balanced_tone': 0.8 if not re.search(r'\b(perfect|awful|terrible|amazing)\b', text.lower()) else 0.6
        }
        
        return np.mean(list(features.values()))
    
    def get_training_data(self, target_column='sentiment_label'):
        """Get training data for specific task"""
        if self.processed_data is None:
            raise ValueError("No data loaded. Call load_amazon_data first.")
        
        if target_column not in self.processed_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")
        
        # Remove rows with missing target
        df_clean = self.processed_data.dropna(subset=[target_column, 'full_review'])
        
        X = df_clean['full_review'].values
        y = df_clean[target_column].values
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def get_credibility_training_data(self):
        """Get training data for credibility model"""
        if self.processed_data is None:
            raise ValueError("No data loaded. Call load_amazon_data first.")
        
        df_clean = self.processed_data.dropna(subset=['credibility_score', 'full_review'])
        
        X = df_clean['full_review'].values
        y = df_clean['credibility_score'].values
        
        return train_test_split(X, y, test_size=0.2, random_state=42)