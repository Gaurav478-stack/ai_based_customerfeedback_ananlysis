import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import joblib

class CredibilityModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_names = []
    
    def extract_credibility_features(self, texts):
        """Extract features that indicate review credibility"""
        features = []
        
        for text in texts:
            text = str(text)
            text_length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Readability features
            avg_sentence_length = word_count / max(sentence_count, 1)
            avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
            
            # Specificity features
            specific_terms = len(re.findall(r'\b(\d+|\d+\.\d+|[A-Z][a-z]+)\b', text))
            capitalized_words = len(re.findall(r'\b[A-Z][a-z]+\b', text))
            
            # Emotional features
            emotional_words = len(re.findall(r'\b(amazing|awful|terrible|excellent|poor|great|bad|good|horrible|wonderful)\b', text.lower()))
            
            # Structural features
            has_details = 1 if any(word in text.lower() for word in ['because', 'since', 'therefore', 'however']) else 0
            has_comparison = 1 if any(word in text.lower() for word in ['than', 'compared', 'versus', 'vs']) else 0
            
            feature_vector = [
                text_length,
                word_count,
                sentence_count,
                avg_sentence_length,
                avg_word_length,
                specific_terms,
                capitalized_words,
                emotional_words,
                has_details,
                has_comparison
            ]
            
            features.append(feature_vector)
        
        self.feature_names = [
            'text_length', 'word_count', 'sentence_count', 'avg_sentence_length',
            'avg_word_length', 'specific_terms', 'capitalized_words', 
            'emotional_words', 'has_details', 'has_comparison'
        ]
        
        return np.array(features)
    
    def train(self, texts, credibility_scores):
        """Train credibility model"""
        # Extract features
        structural_features = self.extract_credibility_features(texts)
        
        # TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(texts)
        
        # Combine features
        from scipy.sparse import hstack
        X = hstack([structural_features, tfidf_features])
        
        # Train model
        self.model.fit(X, credibility_scores)
        
        return self.model
    
    def predict(self, text):
        """Predict credibility score (0-1)"""
        # Extract structural features
        structural_features = self.extract_credibility_features([text])
        
        # TF-IDF features
        tfidf_features = self.vectorizer.transform([text])
        
        # Combine features
        from scipy.sparse import hstack
        X = hstack([structural_features, tfidf_features])
        
        # Predict
        credibility_score = self.model.predict(X)[0]
        
        # Ensure score is between 0 and 1
        credibility_score = max(0, min(1, credibility_score))
        
        return {
            'score': float(credibility_score),
            'interpretation': self.interpret_credibility(credibility_score)
        }
    
    def interpret_credibility(self, score):
        """Interpret credibility score"""
        if score >= 0.8:
            return "Highly Credible"
        elif score >= 0.6:
            return "Credible" 
        elif score >= 0.4:
            return "Moderately Credible"
        else:
            return "Low Credibility"
    
    def save(self, filepath):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.feature_names = data['feature_names']