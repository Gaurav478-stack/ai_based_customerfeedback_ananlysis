import re
import json
from typing import List, Dict, Any
import pandas as pd

class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        text = str(text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\?\!]', '', text)
        return text.strip()
    
    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Extract potential entities from text"""
        # Simple entity extraction - can be enhanced with NER
        entities = []
        
        # Extract product mentions
        product_patterns = [
            r'\b(phone|laptop|tablet|watch|camera|headphone)s?\b',
            r'\b(charger|cable|adapter|case|cover)s?\b',
            r'\b(book|novel|magazine)s?\b'
        ]
        
        for pattern in product_patterns:
            entities.extend(re.findall(pattern, text.lower()))
        
        return list(set(entities))
    
    @staticmethod
    def calculate_readability(text: str) -> float:
        """Calculate simple readability score"""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if len(words) == 0 or len(sentences) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula (higher is more readable)
        readability = 100 - (avg_sentence_length + avg_word_length)
        return max(0, min(100, readability))

class ResultsFormatter:
    """Format analysis results for different outputs"""
    
    @staticmethod
    def to_json(analysis_result: Dict[str, Any]) -> str:
        """Convert analysis result to JSON"""
        return json.dumps(analysis_result, indent=2, ensure_ascii=False)
    
    @staticmethod
    def to_dataframe(analysis_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert multiple analysis results to DataFrame"""
        flattened_results = []
        
        for result in analysis_results:
            flattened = {
                'text': result.get('text', ''),
                'sentiment_label': result.get('sentiment', {}).get('label', ''),
                'sentiment_confidence': result.get('sentiment', {}).get('confidence', 0),
                'credibility_score': result.get('credibility', {}).get('score', 0),
                'credibility_interpretation': result.get('credibility', {}).get('interpretation', ''),
                'topics': ', '.join(result.get('topics', [])),
                'suggestions': ', '.join(result.get('suggestions', [])),
                'priority': result.get('summary', {}).get('priority', ''),
                'action_required': result.get('summary', {}).get('action_required', False)
            }
            flattened_results.append(flattened)
        
        return pd.DataFrame(flattened_results)
    
    @staticmethod
    def generate_report(analysis_results: List[Dict[str, Any]]) -> str:
        """Generate a human-readable report"""
        if not analysis_results:
            return "No analysis results available."
        
        report = []
        report.append("CUSTOMER FEEDBACK ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Summary statistics
        total_feedbacks = len(analysis_results)
        sentiment_counts = {}
        high_priority_count = 0
        
        for result in analysis_results:
            sentiment = result.get('sentiment', {}).get('label', 'unknown')
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            if result.get('summary', {}).get('priority') == 'high':
                high_priority_count += 1
        
        report.append(f"Total Feedbacks Analyzed: {total_feedbacks}")
        report.append(f"Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_feedbacks) * 100
            report.append(f"  - {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        report.append(f"High Priority Items: {high_priority_count}")
        report.append("")
        
        # High priority items
        high_priority_items = [
            result for result in analysis_results 
            if result.get('summary', {}).get('priority') == 'high'
        ]
        
        if high_priority_items:
            report.append("HIGH PRIORITY FEEDBACKS:")
            report.append("-" * 30)
            for i, item in enumerate(high_priority_items[:5], 1):
                report.append(f"{i}. {item.get('text', '')[:100]}...")
                report.append(f"   Sentiment: {item.get('sentiment', {}).get('label', '')}")
                report.append(f"   Topics: {', '.join(item.get('topics', []))}")
                report.append("")
        
        return "\n".join(report)