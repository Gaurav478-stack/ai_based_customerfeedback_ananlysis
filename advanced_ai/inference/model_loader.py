import joblib
import os
from advanced_ai.models.sentiment_model import SentimentModel
from advanced_ai.models.credibility_model import CredibilityModel
from advanced_ai.models.suggestion_model import SuggestionModel
from advanced_ai.models.topic_model import TopicModel

class ModelLoader:
    def __init__(self, models_dir="advanced_ai/data/trained_models"):
        self.models_dir = models_dir
        self.models = {}
    
    def load_all_models(self):
        """Load all trained models"""
        models = {}
        
        # Sentiment Model
        sentiment_path = os.path.join(self.models_dir, "sentiment_model.pkl")
        if os.path.exists(sentiment_path):
            sentiment_model = SentimentModel()
            sentiment_model.load(sentiment_path)
            models['sentiment'] = sentiment_model
            print("✅ Sentiment model loaded")
        else:
            print("❌ Sentiment model not found")
        
        # Credibility Model
        credibility_path = os.path.join(self.models_dir, "credibility_model.pkl")
        if os.path.exists(credibility_path):
            credibility_model = CredibilityModel()
            credibility_model.load(credibility_path)
            models['credibility'] = credibility_model
            print("✅ Credibility model loaded")
        else:
            print("❌ Credibility model not found")
        
        # Suggestion Model
        suggestion_path = os.path.join(self.models_dir, "suggestion_model.pkl")
        if os.path.exists(suggestion_path):
            suggestion_model = SuggestionModel()
            suggestion_model.load(suggestion_path)
            models['suggestion'] = suggestion_model
            print("✅ Suggestion model loaded")
        else:
            print("❌ Suggestion model not found")
        
        # Topic Model
        topic_path = os.path.join(self.models_dir, "topic_model.pkl")
        if os.path.exists(topic_path):
            topic_model = TopicModel()
            topic_model.load(topic_path)
            models['topic'] = topic_model
            print("✅ Topic model loaded")
        else:
            print("❌ Topic model not found")
        
        self.models = models
        return models
    
    def get_model(self, model_name):
        """Get specific model by name"""
        return self.models.get(model_name)