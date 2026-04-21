import nltk
import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_text(text):
    print("Original Text:\n", text, "\n")
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 4. Tokenization
    # Fallback and download if punkt is not available
    try:
        tokens = word_tokenize(text)
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        tokens = word_tokenize(text)
        
    # 5. Remove stop words
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
    tokens = [word for word in tokens if word not in stop_words]
    
    # 6. Lemmatization
    try:
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word) for word in tokens]
        
    return lemmas

if __name__ == "__main__":
    filepath = "sample_data.txt"
    try:
        text_data = load_data(filepath)
        processed_tokens = preprocess_text(text_data)
        print("Processed Tokens:")
        print(processed_tokens)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Make sure it exists in the same directory.")
