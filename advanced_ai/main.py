from advanced_ai.inference.predictor import FeedbackPredictor
from advanced_ai.utils.config import Config
import pandas as pd
import json

def main():
    """Main entry point for customer feedback analyzer"""
    print("🚀 Customer Feedback Analyzer - Advanced AI System")
    
    # Initialize predictor
    predictor = FeedbackPredictor()
    
    # Sample feedback for testing
    sample_feedbacks = [
        "This product is absolutely amazing! The quality exceeded my expectations and delivery was fast.",
        "The item arrived damaged and customer service was unhelpful. Very disappointed.",
        "Average product, does what it says but nothing special. Shipping took longer than expected.",
        "I love this! The features are incredible and it's very user friendly. Highly recommended!",
        "Poor quality material, broke after one week. Would not recommend to anyone."
    ]
    
    print("\n📊 Analyzing Sample Feedbacks:")
    print("=" * 60)
    
    for i, feedback in enumerate(sample_feedbacks, 1):
        print(f"\n{i}. Feedback: {feedback}")
        result = predictor.analyze_feedback(feedback)
        
        print(f"   Sentiment: {result['sentiment']['label']} (Score: {result['sentiment']['confidence']:.3f})")
        print(f"   Credibility: {result['credibility']['score']:.3f}")
        print(f"   Topics: {', '.join(result['topics'])}")
        print(f"   Suggestions: {result['suggestions']}")

if __name__ == "__main__":
    main()