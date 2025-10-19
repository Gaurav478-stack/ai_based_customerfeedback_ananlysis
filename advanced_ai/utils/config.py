import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    SENTIMENT_MODEL_PATH: str = "advanced_ai/data/trained_models/sentiment_model.pkl"
    CREDIBILITY_MODEL_PATH: str = "advanced_ai/data/trained_models/credibility_model.pkl"
    SUGGESTION_MODEL_PATH: str = "advanced_ai/data/trained_models/suggestion_model.pkl"
    TOPIC_MODEL_PATH: str = "advanced_ai/data/trained_models/topic_model.pkl"
    
    # Model parameters
    SENTIMENT_THRESHOLD: float = 0.6
    CREDIBILITY_THRESHOLD: float = 0.5
    MAX_SUGGESTIONS: int = 3
    NUM_TOPICS: int = 8

@dataclass
class DataConfig:
    """Configuration for data processing"""
    RAW_DATA_PATH: str = "advanced_ai/data/raw/amazon.xlsx"
    PROCESSED_DATA_PATH: str = "advanced_ai/data/processed/cleaned_data.csv"
    MAX_TEXT_LENGTH: int = 1000
    MIN_TEXT_LENGTH: int = 10

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from environment or config file"""
        return {
            'debug': os.getenv('DEBUG', 'False').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'cache_results': True,
            'max_batch_size': 100
        }
    
    def get_model_paths(self) -> Dict[str, str]:
        """Get all model file paths"""
        return {
            'sentiment': self.model.SENTIMENT_MODEL_PATH,
            'credibility': self.model.CREDIBILITY_MODEL_PATH,
            'suggestion': self.model.SUGGESTION_MODEL_PATH,
            'topic': self.model.TOPIC_MODEL_PATH
        }