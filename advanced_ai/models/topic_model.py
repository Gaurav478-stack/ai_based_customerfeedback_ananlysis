import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import re
import joblib
from collections import Counter

class TopicModel:
    def __init__(self, n_topics=8):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.lda = None
        self.nmf = None
        self.kmeans = None
        self.feature_names = []
        self.topic_names = {}
    
    def preprocess_texts(self, texts):
        """Preprocess texts for topic modeling"""
        processed_texts = []
        for text in texts:
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            processed_texts.append(text)
        return processed_texts
    
    def train(self, texts):
        """Train multiple topic models"""
        processed_texts = self.preprocess_texts(texts)
        
        # Create document-term matrix
        dtm = self.vectorizer.fit_transform(processed_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Train LDA
        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=10
        )
        lda_topics = self.lda.fit_transform(dtm)
        
        # Train NMF
        self.nmf = NMF(
            n_components=self.n_topics,
            random_state=42,
            max_iter=200
        )
        nmf_topics = self.nmf.fit_transform(dtm)
        
        # Train K-means on TF-IDF features
        self.kmeans = KMeans(n_clusters=self.n_topics, random_state=42)
        kmeans_labels = self.kmeans.fit_predict(dtm)
        
        # Generate topic names
        self.generate_topic_names()
        
        return {
            'lda_topics': lda_topics,
            'nmf_topics': nmf_topics,
            'kmeans_labels': kmeans_labels
        }
    
    def generate_topic_names(self):
        """Generate descriptive names for topics"""
        if self.lda is None:
            return
        
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            
            # Create topic name based on top words
            if any(word in top_words for word in ['quality', 'material', 'durable']):
                topic_name = "Product Quality"
            elif any(word in top_words for word in ['price', 'cost', 'expensive']):
                topic_name = "Pricing"
            elif any(word in top_words for word in ['delivery', 'shipping', 'arrived']):
                topic_name = "Delivery Experience"
            elif any(word in top_words for word in ['service', 'support', 'help']):
                topic_name = "Customer Service"
            elif any(word in top_words for word in ['easy', 'use', 'user']):
                topic_name = "Usability"
            elif any(word in top_words for word in ['feature', 'function', 'performance']):
                topic_name = "Features & Performance"
            elif any(word in top_words for word in ['love', 'great', 'excellent']):
                topic_name = "Positive Experience"
            elif any(word in top_words for word in ['bad', 'poor', 'terrible']):
                topic_name = "Negative Experience"
            else:
                topic_name = f"Topic {topic_idx + 1}"
            
            self.topic_names[topic_idx] = topic_name
    
    def predict_topics(self, text):
        """Predict topics for a single text"""
        if self.lda is None:
            return ["General Feedback"]
        
        processed_text = self.preprocess_texts([text])[0]
        dtm = self.vectorizer.transform([processed_text])
        
        # Get LDA topic distribution
        lda_topic_dist = self.lda.transform(dtm)[0]
        
        # Get top topics
        top_topic_indices = lda_topic_dist.argsort()[-3:][::-1]
        top_topics = []
        
        for idx in top_topic_indices:
            if lda_topic_dist[idx] > 0.1:  # Threshold
                topic_name = self.topic_names.get(idx, f"Topic {idx}")
                top_topics.append({
                    'topic': topic_name,
                    'confidence': float(lda_topic_dist[idx])
                })
        
        # If no strong topics, use keyword matching
        if not top_topics:
            return self.fallback_topic_detection(text)
        
        # Return just topic names for simplicity
        return [topic['topic'] for topic in top_topics[:2]]  # Top 2 topics
    
    def fallback_topic_detection(self, text):
        """Fallback topic detection using keyword matching"""
        text_lower = text.lower()
        topics = []
        
        topic_keywords = {
            "Product Quality": ['quality', 'material', 'durable', 'break', 'last'],
            "Pricing": ['price', 'cost', 'expensive', 'cheap', 'worth'],
            "Delivery": ['delivery', 'shipping', 'arrived', 'package', 'delivered'],
            "Customer Service": ['service', 'support', 'help', 'response', 'contact'],
            "Usability": ['easy', 'use', 'user', 'interface', 'simple'],
            "Features": ['feature', 'function', 'performance', 'speed', 'battery']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ["General Feedback"]
    
    def get_topic_keywords(self, topic_idx, n_words=10):
        """Get top keywords for a topic"""
        if self.lda is None:
            return []
        
        topic = self.lda.components_[topic_idx]
        top_words_idx = topic.argsort()[-n_words:][::-1]
        return [self.feature_names[i] for i in top_words_idx]
    
    def save(self, filepath):
        """Save topic models"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'lda': self.lda,
            'nmf': self.nmf,
            'kmeans': self.kmeans,
            'topic_names': self.topic_names,
            'feature_names': self.feature_names
        }, filepath)
    
    def load(self, filepath):
        """Load topic models"""
        data = joblib.load(filepath)
        self.vectorizer = data['vectorizer']
        self.lda = data['lda']
        self.nmf = data['nmf']
        self.kmeans = data['kmeans']
        self.topic_names = data['topic_names']
        self.feature_names = data['feature_names']