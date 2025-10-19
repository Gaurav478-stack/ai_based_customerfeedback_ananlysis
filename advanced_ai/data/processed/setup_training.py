import os
import nltk
import pandas as pd
from advanced_ai.training.data_preprocessor import DataPreprocessor

def setup_environment():
    """Setup the environment for training"""
    print("🔧 Setting up training environment...")
    
    # Create necessary directories
    directories = [
        'advanced_ai/data/raw',
        'advanced_ai/data/processed', 
        'advanced_ai/data/trained_models',
        'results/training'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Download NLTK data
    print("📥 Downloading NLTK data...")
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except:
        print("⚠️  NLTK download failed, but we can continue")
    
    print("🎉 Environment setup completed!")

def check_data_file():
    """Check if data file exists and show basic info"""
    data_path = "advanced_ai/data/raw/amazon.xlsx"
    
    if os.path.exists(data_path):
        print(f"✅ Data file found: {data_path}")
        try:
            # Try to load and show basic info
            df = pd.read_excel(data_path)
            print(f"   📊 Data shape: {df.shape}")
            print(f"   📝 Columns: {list(df.columns)}")
            
            # Show sample of review content
            if 'review_content' in df.columns:
                sample_review = df['review_content'].iloc[0] if len(df) > 0 else "No reviews"
                print(f"   📄 Sample review: {str(sample_review)[:100]}...")
            
            return True
        except Exception as e:
            print(f"❌ Error reading data file: {e}")
            return False
    else:
        print(f"❌ Data file not found: {data_path}")
        print("   Please place your 'amazon.xlsx' file in advanced_ai/data/raw/")
        return False

if __name__ == "__main__":
    setup_environment()
    check_data_file()