from flask import Flask, request, jsonify
from textblob import TextBlob
import nltk
import pandas as pd
import re
import torch
import numpy as np
import sys
import os

# Add advanced_ai to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'advanced_ai'))

# Download NLTK data
nltk.download('punkt', quiet=True)

app = Flask(__name__)

class SimpleAnalyzer:
    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
            
        return {
            'sentiment_score': polarity,
            'sentiment_label': label
        }
    
    def calculate_credibility(self, text, rating):
        # Simple credibility based on text length and rating consistency
        text_length_score = min(len(text) / 200, 1.0)
        
        # Check if sentiment matches rating
        sentiment = self.analyze_sentiment(text)
        expected_sentiment = (rating - 3) / 2
        consistency = 1.0 - abs(sentiment['sentiment_score'] - expected_sentiment) / 2
        
        credibility = (text_length_score + consistency) / 2
        return max(0.1, min(1.0, credibility))

# Initialize simple analyzer
simple_analyzer = SimpleAnalyzer()

# Try to initialize advanced analyzer (will use simple if not available)
try:
    from advanced_inference import AdvancedAnalyzer
    advanced_analyzer = AdvancedAnalyzer()
    ADVANCED_AI_AVAILABLE = True
    print("Advanced AI models loaded successfully!")
except ImportError as e:
    print(f"Advanced AI not available: {e}")
    print("Falling back to simple analyzer")
    ADVANCED_AI_AVAILABLE = False
except Exception as e:
    print(f"Error loading advanced AI: {e}")
    print("Falling back to simple analyzer")
    ADVANCED_AI_AVAILABLE = False

@app.route('/health', methods=['GET'])
def health_check():
    ai_status = "advanced" if ADVANCED_AI_AVAILABLE else "simple"
    return jsonify({
        'status': 'healthy', 
        'ai_capability': ai_status,
        'service': 'Feedback Analyzer AI'
    })

@app.route('/analyze', methods=['POST'])
def analyze_review():
    """Original endpoint - maintains backward compatibility"""
    try:
        data = request.get_json()
        
        review_text = data.get('review_text', '')
        rating = data.get('rating', 3)
        
        sentiment = simple_analyzer.analyze_sentiment(review_text)
        credibility = simple_analyzer.calculate_credibility(review_text, rating)
        
        return jsonify({
            'review_id': data.get('review_id'),
            'sentiment_score': sentiment['sentiment_score'],
            'sentiment_label': sentiment['sentiment_label'],
            'credibility_score': credibility,
            'analysis_type': 'simple'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/advanced_analyze', methods=['POST'])
def advanced_analyze():
    """Advanced AI analysis endpoint"""
    try:
        if not ADVANCED_AI_AVAILABLE:
            return jsonify({
                'error': 'Advanced AI not available. Using simple analyzer.',
                'fallback': True
            }), 503
        
        data = request.get_json()
        
        review_text = data.get('review_text', '')
        rating = data.get('rating', 3)
        reviewer_name = data.get('reviewer_name', '')
        verified_purchase = data.get('verified_purchase', False)
        helpful_votes = data.get('helpful_votes', 0)
        total_votes = data.get('total_votes', 0)
        
        # Perform advanced analysis
        result = advanced_analyzer.analyze_review(
            review_text=review_text,
            rating=rating,
            reviewer_name=reviewer_name,
            verified_purchase=verified_purchase,
            helpful_votes=helpful_votes,
            total_votes=total_votes
        )
        
        result['analysis_type'] = 'advanced'
        return jsonify(result)
        
    except Exception as e:
        # Fallback to simple analyzer if advanced fails
        print(f"Advanced analysis failed, using fallback: {e}")
        try:
            data = request.get_json()
            review_text = data.get('review_text', '')
            rating = data.get('rating', 3)
            
            sentiment = simple_analyzer.analyze_sentiment(review_text)
            credibility = simple_analyzer.calculate_credibility(review_text, rating)
            
            return jsonify({
                'review_id': data.get('review_id'),
                'sentiment_score': sentiment['sentiment_score'],
                'sentiment_label': sentiment['sentiment_label'],
                'credibility_score': credibility,
                'analysis_type': 'simple_fallback',
                'warning': 'Advanced AI failed, used simple analyzer'
            })
        except Exception as fallback_error:
            return jsonify({'error': str(fallback_error)}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Batch analysis with simple analyzer"""
    try:
        data = request.get_json()
        reviews = data.get('reviews', [])
        
        results = []
        for review in reviews:
            sentiment = simple_analyzer.analyze_sentiment(review['review_text'])
            credibility = simple_analyzer.calculate_credibility(review['review_text'], review.get('rating', 3))
            
            results.append({
                'review_id': review.get('review_id'),
                'sentiment_score': sentiment['sentiment_score'],
                'sentiment_label': sentiment['sentiment_label'],
                'credibility_score': credibility,
                'analysis_type': 'simple'
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_advanced_analyze', methods=['POST'])
def batch_advanced_analyze():
    """Batch advanced analysis"""
    try:
        if not ADVANCED_AI_AVAILABLE:
            return jsonify({
                'error': 'Advanced AI not available',
                'fallback': True
            }), 503
        
        data = request.get_json()
        reviews = data.get('reviews', [])
        
        results = []
        successful = 0
        failed = 0
        
        for review in reviews:
            try:
                result = advanced_analyzer.analyze_review(
                    review_text=review.get('review_text', ''),
                    rating=review.get('rating', 3),
                    reviewer_name=review.get('reviewer_name', ''),
                    verified_purchase=review.get('verified_purchase', False),
                    helpful_votes=review.get('helpful_votes', 0),
                    total_votes=review.get('total_votes', 0)
                )
                result['analysis_type'] = 'advanced'
                results.append(result)
                successful += 1
            except Exception as e:
                # Fallback to simple analysis for failed items
                print(f"Advanced analysis failed for review {review.get('review_id')}: {e}")
                sentiment = simple_analyzer.analyze_sentiment(review.get('review_text', ''))
                credibility = simple_analyzer.calculate_credibility(review.get('review_text', ''), review.get('rating', 3))
                
                results.append({
                    'review_id': review.get('review_id'),
                    'sentiment_score': sentiment['sentiment_score'],
                    'sentiment_label': sentiment['sentiment_label'],
                    'credibility_score': credibility,
                    'analysis_type': 'simple_fallback',
                    'warning': 'Advanced AI failed for this review'
                })
                failed += 1
        
        return jsonify({
            'results': results,
            'summary': {
                'total_processed': len(reviews),
                'successful_advanced': successful,
                'fallback_to_simple': failed
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_smart', methods=['POST'])
def analyze_smart():
    """Smart endpoint that uses advanced AI if available, otherwise falls back to simple"""
    try:
        if ADVANCED_AI_AVAILABLE:
            # Use advanced analyzer
            return advanced_analyze()
        else:
            # Use simple analyzer
            return analyze_review()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/capabilities', methods=['GET'])
def get_capabilities():
    """Get information about available AI capabilities"""
    return jsonify({
        'advanced_ai_available': ADVANCED_AI_AVAILABLE,
        'endpoints': {
            '/analyze': 'Simple sentiment analysis (always available)',
            '/advanced_analyze': 'Advanced AI analysis (if models are trained)',
            '/analyze_smart': 'Auto-selects best available analyzer',
            '/batch_analyze': 'Batch simple analysis',
            '/batch_advanced_analyze': 'Batch advanced analysis'
        },
        'features': {
            'sentiment_analysis': True,
            'credibility_scoring': True,
            'advanced_sentiment': ADVANCED_AI_AVAILABLE,
            'suggestion_generation': ADVANCED_AI_AVAILABLE,
            'topic_analysis': ADVANCED_AI_AVAILABLE
        }
    })

if __name__ == '__main__':
    print("Starting Feedback Analyzer AI Service...")
    print(f"Advanced AI Available: {ADVANCED_AI_AVAILABLE}")
    print("Available endpoints:")
    print("  GET  /health - Service health check")
    print("  POST /analyze - Simple sentiment analysis")
    print("  POST /advanced_analyze - Advanced AI analysis")
    print("  POST /analyze_smart - Auto-selects best analyzer")
    print("  POST /batch_analyze - Batch simple analysis")
    print("  POST /batch_advanced_analyze - Batch advanced analysis")
    print("  GET  /capabilities - List available features")
    
    app.run(host='0.0.0.0', port=5000, debug=False)