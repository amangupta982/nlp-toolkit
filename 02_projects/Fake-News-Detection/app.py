import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def perform_eda(df):
    plt.figure(figsize=(12, 5))
    
    # 1. Class Distribution Plot
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='label', palette='Set2')
    plt.title('Distribution of Fake vs Real News')
    plt.xticks(ticks=[0, 1], labels=['Real (0)', 'Fake (1)'])
    plt.xlabel('News Type')
    plt.ylabel('Count')
    
    # 2. Text Length Distribution
    # Calculate length of each text
    df['text_length'] = df['text'].apply(len)
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='text_length', hue='label', kde=True, palette='Set2', bins=10)
    plt.title('Text Length Distribution by Class')
    plt.xlabel('Length of Text (characters)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('eda_graphs.png')
    print("EDA graphs saved as 'eda_graphs.png'")
    plt.close()

def train_and_evaluate(df):
    # Features and Labels
    X = df['text']
    y = df['label']
    
    # Split Data
    # For tiny datasets, a simple split ensures at least some data in train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # TF-IDF Vectorization
    try:
        nltk_stopwords = stopwords.words('english')
    except Exception:
        nltk_stopwords = 'english'

    vectorizer = TfidfVectorizer(
        tokenizer=word_tokenize, 
        stop_words=nltk_stopwords,
        token_pattern=None,
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model Training
    # Increased C parameter to improve performance on small training set
    model = LogisticRegression(C=10.0, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Accuracy: {acc * 100:.2f}% ---")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)'], zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()

if __name__ == "__main__":
    filepath = "fake_news_dataset.csv"
    try:
        df = load_data(filepath)
        print("Data Loaded Successfully. Total records:", len(df))
        
        # 1. Perform Exploratory Data Analysis
        perform_eda(df)
        
        # 2. Train Model and output Confusion Matrix
        train_and_evaluate(df)
        print("\nAll tasks completed successfully. Check the directory for the generated graphs.")
        
    except FileNotFoundError:
        print(f"Error: dataset '{filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
