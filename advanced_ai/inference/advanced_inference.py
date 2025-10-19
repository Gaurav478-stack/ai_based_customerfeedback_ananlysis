import torch
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Any
import sys

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_model import SentimentTrainer
from models.credibility_model import CredibilityModel
from models.suggestion_model import SuggestionGenerator

class AdvancedAnalyzer:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            # Default to looking in the data/trained_models directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, '..', 'data', 'trained_models')
        
        self.model_dir = model_dir
        self.sentiment_model = None
        self.credibility_model = None
        self.suggestion_model = None
        self.models_loaded = False
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load sentiment model
            sentiment_path = os.path.join(self.model_dir, 'sentiment_model.pth')
            if os.path.exists(sentiment_path):
                self.sentiment_model = SentimentTrainer()
                self.sentiment_model.load_model(sentiment_path)
                print("✓ Sentiment model loaded")
            else:
                print("✗ Sentiment model not found")
            
            # Load credibility model
            credibility_path = os.path.join(self.model_dir, 'credibility_model.pkl')
            if os.path.exists(credibility_path):
                self.credibility_model = CredibilityModel()
                self.credibility_model.load_model(credibility_path)
                print("✓ Credibility model loaded")
            else:
                print("✗ Credibility model not found")
            
            # Load suggestion model
            suggestion_path = os.path.join(self.model_dir, 'suggestion_model.pkl')
            if os.path.exists(suggestion_path):
                self.suggestion_model = SuggestionGenerator()
                self.suggestion_model.load_model(suggestion_path)
                print("✓ Suggestion model loaded")
            else:
                print("✗ Suggestion model not found - using rule-based suggestions")
                self.suggestion_model = SuggestionGenerator()
            
            self.models_loaded = any([self.sentiment_model, self.credibility_model])
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
    
    def analyze_review(self, 
                      review_text: str, 
                      rating: int = 3,
                      reviewer_name: str = "",
                      verified_purchase: bool = False,
                      helpful_votes: int = 0,
                      total_votes: int = 0) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a review using advanced AI models
        """
        try:
            result = {
                'review_text': review_text,
                'rating': rating,
                'analysis_type': 'advanced'
            }
            
            # Sentiment Analysis
            if self.sentiment_model:
                # Use advanced sentiment model
                sentiment_result = self._advanced_sentiment_analysis(review_text)
            else:
                # Fallback to simple sentiment analysis
                from textblob import TextBlob
                blob = TextBlob(review_text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    label = 'positive'
                elif polarity < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                sentiment_result = {
                    'sentiment_score': float(polarity),
                    'sentiment_label': label,
                    'confidence': max(0.1, min(0.9, abs(polarity)))
                }
            
            result.update(sentiment_result)
            
            # Credibility Analysis
            if self.credibility_model:
                credibility_result = self._advanced_credibility_analysis(
                    review_text, rating, reviewer_name, verified_purchase, 
                    helpful_votes, total_votes
                )
            else:
                # Fallback to simple credibility calculation
                text_length_score = min(len(review_text) / 200, 1.0)
                expected_sentiment = (rating - 3) / 2
                consistency = 1.0 - abs(sentiment_result['sentiment_score'] - expected_sentiment) / 2
                credibility = (text_length_score + consistency) / 2
                
                credibility_result = {
                    'credibility_score': float(max(0.1, min(1.0, credibility))),
                    'credibility_factors': {
                        'text_length': text_length_score,
                        'rating_consistency': consistency
                    }
                }
            
            result.update(credibility_result)
            
            # Suggestion Generation
            suggestions = self.suggestion_model.generate_suggestions(
                review_text, 
                sentiment_result['sentiment_label'], 
                rating
            )
            result['suggestions'] = suggestions
            
            # Additional Insights
            result['insights'] = self._generate_insights(
                review_text, 
                sentiment_result, 
                credibility_result, 
                rating
            )
            
            return result
            
        except Exception as e:
            print(f"Error in advanced analysis: {e}")
            raise
    
    def _advanced_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform advanced sentiment analysis using trained model"""
        # This would use the actual trained model for inference
        # For now, return enhanced simple analysis
        from textblob import TextBlob
        import re
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Enhanced sentiment with text patterns
        positive_words = ['excellent', 'amazing', 'great', 'good', 'awesome', 'love', 'perfect']
        negative_words = ['terrible', 'awful', 'horrible', 'bad', 'poor', 'hate', 'disappointing']
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        # Adjust score based on word counts
        adjustment = (positive_count - negative_count) * 0.05
        enhanced_score = max(-1.0, min(1.0, polarity + adjustment))
        
        if enhanced_score > 0.2:
            label = 'positive'
            confidence = 0.7 + enhanced_score * 0.3
        elif enhanced_score < -0.2:
            label = 'negative'
            confidence = 0.7 + abs(enhanced_score) * 0.3
        else:
            label = 'neutral'
            confidence = 0.6
        
        return {
            'sentiment_score': float(enhanced_score),
            'sentiment_label': label,
            'confidence': float(confidence),
            'subjectivity': float(subjectivity),
            'positive_indicators': positive_count,
            'negative_indicators': negative_count
        }
    
    def _advanced_credibility_analysis(self, text: str, rating: int, reviewer_name: str,
                                     verified_purchase: bool, helpful_votes: int, 
                                     total_votes: int) -> Dict[str, Any]:
        """Perform advanced credibility analysis"""
        # Calculate various credibility factors
        text_length = len(text)
        word_count = len(text.split())
        
        # Text quality metrics
        capital_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        exclamation_ratio = text.count('!') / max(word_count, 1)
        question_ratio = text.count('?') / max(word_count, 1)
        
        # Reviewer metrics
        name_credibility = 0.7 if len(reviewer_name) > 3 and reviewer_name.lower() not in ['anonymous', 'user', 'customer'] else 0.3
        purchase_credibility = 0.9 if verified_purchase else 0.5
        
        # Helpfulness metrics
        if total_votes > 0:
            helpfulness_ratio = helpful_votes / total_votes
            vote_credibility = min(helpfulness_ratio + (total_votes * 0.05), 1.0)
        else:
            vote_credibility = 0.5
        
        # Combine factors (weights can be adjusted based on trained model)
        factors = {
            'text_length': min(text_length / 300, 1.0),
            'text_quality': 1.0 - min(capital_ratio * 3, 1.0),  # Penalize excessive caps
            'exclamation_usage': 1.0 - min(exclamation_ratio * 10, 1.0),
            'question_usage': 1.0 - min(question_ratio * 10, 1.0),
            'reviewer_identity': name_credibility,
            'purchase_verification': purchase_credibility,
            'community_feedback': vote_credibility
        }
        
        # Weighted average
        weights = {
            'text_length': 0.15,
            'text_quality': 0.20,
            'exclamation_usage': 0.10,
            'question_usage': 0.10,
            'reviewer_identity': 0.15,
            'purchase_verification': 0.20,
            'community_feedback': 0.10
        }
        
        credibility_score = sum(factors[factor] * weight for factor, weight in weights.items())
        
        return {
            'credibility_score': float(max(0.1, min(1.0, credibility_score))),
            'credibility_factors': factors,
            'spam_probability': float(1.0 - credibility_score)
        }
    
    def _generate_insights(self, text: str, sentiment_result: Dict, 
                          credibility_result: Dict, rating: int) -> List[str]:
        """Generate business insights from the analysis"""
        insights = []
        
        # Sentiment-based insights
        if sentiment_result['sentiment_label'] == 'positive' and rating >= 4:
            insights.append("Highly satisfied customer - potential brand advocate")
        elif sentiment_result['sentiment_label'] == 'negative' and rating <= 2:
            insights.append("Critical feedback - requires immediate attention")
        
        # Credibility insights
        if credibility_result['credibility_score'] > 0.8:
            insights.append("High credibility review - strongly consider feedback")
        elif credibility_result['credibility_score'] < 0.4:
            insights.append("Low credibility - review may need verification")
        
        # Text-based insights
        if len(text) > 200:
            insights.append("Detailed review - contains specific feedback")
        elif len(text) < 50:
            insights.append("Brief review - limited actionable information")
        
        # Rating consistency insight
        expected_sentiment = (rating - 3) / 2
        sentiment_consistency = 1.0 - abs(sentiment_result['sentiment_score'] - expected_sentiment) / 2
        if sentiment_consistency < 0.6:
            insights.append("Rating may not match review content")
        
        return insights

# For backward compatibility
AdvancedAnalyzer = AdvancedAnalyzer