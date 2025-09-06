from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class TFIDFExtractor:
    """TF-IDF vectorizer wrapper with fit/transform/save/load."""
    def __init__(self, params):
        self.vectorizer = TfidfVectorizer(**params)  # e.g., ngrams for phrases
    
    def fit(self, texts):
        self.vectorizer.fit(texts)
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    
    def save(self, path):
        joblib.dump(self.vectorizer, path)
    
    def load(self, path):
        self.vectorizer = joblib.load(path)