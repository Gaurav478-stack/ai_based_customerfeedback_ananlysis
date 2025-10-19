import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.models = {}
        self.sia = SentimentIntensityAnalyzer()
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_features(self, texts):
        """Extract multiple feature types"""
        # TF-IDF features
        tfidf_features = self.vectorizer.transform(texts)
        
        # Sentiment lexicon features
        lexicon_features = []
        for text in texts:
            scores = self.sia.polarity_scores(text)
            lexicon_features.append([
                scores['pos'],
                scores['neg'], 
                scores['neu'],
                scores['compound']
            ])
        
        lexicon_features = np.array(lexicon_features)
        
        # Combine features
        from scipy.sparse import hstack
        features = hstack([tfidf_features, lexicon_features])
        
        return features
    
    def train(self, X, y):
        """Train multiple sentiment models"""
        # Preprocess texts
        X_processed = [self.preprocess_text(text) for text in X]
        
        # Fit vectorizer
        self.vectorizer.fit(X_processed)
        
        # Extract features
        X_features = self.extract_features(X_processed)
        
        # Define models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_features, y)
    
    def predict(self, text):
        """Predict sentiment with confidence scores"""
        processed_text = self.preprocess_text(text)
        features = self.extract_features([processed_text])
        
        predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                prediction = model.predict(features)[0]
                confidence = max(proba)
            else:
                prediction = model.predict(features)[0]
                confidence = 1.0
            
            predictions[name] = {
                'prediction': prediction,
                'confidence': confidence
            }
        
        # Ensemble prediction (majority voting)
        final_prediction = max(set([p['prediction'] for p in predictions.values()]), 
                              key=[p['prediction'] for p in predictions.values()].count)
        
        avg_confidence = np.mean([p['confidence'] for p in predictions.values()])
        
        return {
            'label': final_prediction,
            'confidence': avg_confidence,
            'model_predictions': predictions
        }
    
    def save(self, filepath):
        """Save model and vectorizer"""
        joblib.dump({
            'models': self.models,
            'vectorizer': self.vectorizer
        }, filepath)
    
    def load(self, filepath):
        """Load model and vectorizer"""
        data = joblib.load(filepath)
        self.models = data['models']
        self.vectorizer = data['vectorizer']