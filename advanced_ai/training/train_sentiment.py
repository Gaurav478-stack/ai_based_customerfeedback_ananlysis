from advanced_ai.models.sentiment_model import SentimentModel
from advanced_ai.training.data_preprocessor import DataPreprocessor
import joblib
import os

class SentimentTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = SentimentModel()
    
    def train_from_file(self, data_filepath, model_save_path):
        """Train sentiment model from data file"""
        print("Loading and preprocessing data...")
        self.preprocessor.load_amazon_data(data_filepath)
        
        print("Preparing training data...")
        X_train, X_test, y_train, y_test = self.preprocessor.get_training_data('sentiment_label')
        
        print(f"Training on {len(X_train)} samples...")
        self.model.train(X_train, y_train)
        
        # Evaluate on test set
        print("Evaluating model...")
        accuracy = self.evaluate_model(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.3f}")
        
        # Save model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        self.model.save(model_save_path)
        print(f"Model saved to: {model_save_path}")
        
        return accuracy
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        correct = 0
        total = len(X_test)
        
        for text, true_label in zip(X_test, y_test):
            prediction = self.model.predict(text)
            if prediction['label'] == true_label:
                correct += 1
        
        return correct / total

def main():
    """Main training function"""
    trainer = SentimentTrainer()
    
    # Update this path to your data file
    data_file = "advanced_ai/data/raw/amazon.xlsx"
    model_save_path = "advanced_ai/data/trained_models/sentiment_model.pkl"
    
    if os.path.exists(data_file):
        accuracy = trainer.train_from_file(data_file, model_save_path)
        print(f"✅ Sentiment model training completed! Accuracy: {accuracy:.3f}")
    else:
        print(f"❌ Data file not found: {data_file}")
        print("Please update the data_file path in train_sentiment.py")

if __name__ == "__main__":
    main()