import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import re
import joblib

class SuggestionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=4000, stop_words='english')
        self.mlb = MultiLabelBinarizer()
        self.model = None
        self.suggestion_categories = [
            'improve_quality',
            'enhance_features', 
            'better_packaging',
            'faster_delivery',
            'price_adjustment',
            'customer_service',
            'user_experience',
            'product_variety'
        ]
    
    def extract_suggestions_from_text(self, text):
        """Extract potential suggestions from text using rule-based approach"""
        text_lower = text.lower()
        suggestions = []
        
        # Rule-based suggestion extraction
        suggestion_patterns = {
            'improve_quality': [
                r'better quality', r'improve quality', r'poor quality', r'low quality',
                r'break(s|ing)? easily', r'not durable', r'cheap material'
            ],
            'enhance_features': [
                r'more features', r'add feature', r'missing feature', 
                r'should have', r'would be better if', r'include'
            ],
            'better_packaging': [
                r'better packaging', r'package damaged', r'arrived broken',
                r'poor packaging', r'packaging issue'
            ],
            'faster_delivery': [
                r'faster delivery', r'shipping slow', r'took long', 
                r'delivery time', r'late delivery'
            ],
            'price_adjustment': [
                r'too expensive', r'overpriced', r'price high', 
                r'cheaper', r'reduce price', r'costly'
            ],
            'customer_service': [
                r'customer service', r'support', r'helpful', 
                r'response time', r'contact us'
            ],
            'user_experience': [
                r'hard to use', r'complicated', r'user friendly',
                r'easy to use', r'difficult', r'simple'
            ],
            'product_variety': [
                r'more colors', r'more sizes', r'more options',
                r'variety', r'different types'
            ]
        }
        
        for category, patterns in suggestion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    suggestions.append(category)
                    break
        
        return list(set(suggestions))
    
    def train(self, texts, suggestions_list=None):
        """Train suggestion classification model"""
        if suggestions_list is None:
            # Auto-extract suggestions from text
            suggestions_list = [self.extract_suggestions_from_text(text) for text in texts]
        
        # Prepare labels
        y = self.mlb.fit_transform(suggestions_list)
        
        # Prepare features
        X = self.vectorizer.fit_transform(texts)
        
        # Train multi-label classifier
        self.model = OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000))
        self.model.fit(X, y)
        
        return self.model
    
    def predict(self, text):
        """Predict suggestions from text"""
        if self.model is None:
            # Fallback to rule-based approach
            return self.extract_suggestions_from_text(text)
        
        # Transform text
        X = self.vectorizer.transform([text])
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        # Get top suggestions
        threshold = 0.3
        top_indices = np.where(probabilities > threshold)[0]
        
        if len(top_indices) == 0:
            # Fallback to rule-based if no confident predictions
            return self.extract_suggestions_from_text(text)
        
        suggestions = []
        for idx in top_indices:
            category = self.mlb.classes_[idx]
            confidence = probabilities[idx]
            suggestions.append({
                'category': category,
                'confidence': confidence,
                'suggestion': self.generate_specific_suggestion(category, text)
            })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suggestions
    
    def generate_specific_suggestion(self, category, original_text):
        """Generate specific suggestion based on category and text"""
        suggestions_map = {
            'improve_quality': "Consider improving product quality and durability based on customer feedback",
            'enhance_features': "Evaluate adding requested features to enhance product functionality",
            'better_packaging': "Review packaging process to prevent damage during shipping",
            'faster_delivery': "Optimize delivery process to reduce shipping times",
            'price_adjustment': "Consider pricing strategy based on customer value perception",
            'customer_service': "Enhance customer support and response systems",
            'user_experience': "Improve product usability and user interface design",
            'product_variety': "Expand product options to meet diverse customer needs"
        }
        
        return suggestions_map.get(category, "Consider customer feedback for product improvements")
    
    def save(self, filepath):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'mlb': self.mlb
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.mlb = data['mlb']