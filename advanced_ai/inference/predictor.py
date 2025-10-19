from advanced_ai.inference.model_loader import ModelLoader
import numpy as np

class FeedbackPredictor:
    def __init__(self, models_dir="advanced_ai/data/trained_models"):
        self.model_loader = ModelLoader(models_dir)
        self.models = self.model_loader.load_all_models()
    
    def analyze_feedback(self, text):
        """Comprehensive analysis of customer feedback"""
        if not text or len(text.strip()) < 10:
            return self._empty_response()
        
        analysis = {
            'text': text,
            'sentiment': self._analyze_sentiment(text),
            'credibility': self._analyze_credibility(text),
            'topics': self._analyze_topics(text),
            'suggestions': self._analyze_suggestions(text),
            'summary': {}
        }
        
        # Create summary
        analysis['summary'] = self._create_summary(analysis)
        
        return analysis
    
    def _analyze_sentiment(self, text):
        """Analyze sentiment of the text"""
        sentiment_model = self.models.get('sentiment')
        if sentiment_model:
            return sentiment_model.predict(text)
        else:
            # Fallback sentiment analysis
            return self._fallback_sentiment(text)
    
    def _analyze_credibility(self, text):
        """Analyze credibility of the feedback"""
        credibility_model = self.models.get('credibility')
        if credibility_model:
            return credibility_model.predict(text)
        else:
            return {'score': 0.7, 'interpretation': 'Moderately Credible'}
    
    def _analyze_topics(self, text):
        """Extract topics from the feedback"""
        topic_model = self.models.get('topic')
        if topic_model:
            return topic_model.predict_topics(text)
        else:
            return ['General Feedback']
    
    def _analyze_suggestions(self, text):
        """Extract suggestions from the feedback"""
        suggestion_model = self.models.get('suggestion')
        if suggestion_model:
            suggestions = suggestion_model.predict(text)
            # Return just the suggestion texts
            if isinstance(suggestions, list) and suggestions and isinstance(suggestions[0], dict):
                return [s['suggestion'] for s in suggestions[:2]]  # Top 2 suggestions
            return suggestions
        else:
            return ["Analyze feedback for improvement opportunities"]
    
    def _fallback_sentiment(self, text):
        """Fallback sentiment analysis using simple rules"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'nice', 'awesome']
        negative_words = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'disappointed', 'waste']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'label': 'positive', 'confidence': 0.7}
        elif negative_count > positive_count:
            return {'label': 'negative', 'confidence': 0.7}
        else:
            return {'label': 'neutral', 'confidence': 0.6}
    
    def _create_summary(self, analysis):
        """Create a summary of the analysis"""
        sentiment = analysis['sentiment']['label']
        credibility = analysis['credibility']['score']
        
        summary = {
            'overall_sentiment': sentiment,
            'credibility_level': analysis['credibility']['interpretation'],
            'key_topics': analysis['topics'],
            'action_required': credibility > 0.6 and sentiment == 'negative',
            'priority': 'high' if (credibility > 0.7 and sentiment == 'negative') else 'medium'
        }
        
        return summary
    
    def _empty_response(self):
        """Return empty response for invalid input"""
        return {
            'text': '',
            'sentiment': {'label': 'neutral', 'confidence': 0.0},
            'credibility': {'score': 0.0, 'interpretation': 'Unknown'},
            'topics': [],
            'suggestions': [],
            'summary': {}
        }
    
    def batch_analyze(self, texts):
        """Analyze multiple texts at once"""
        return [self.analyze_feedback(text) for text in texts]