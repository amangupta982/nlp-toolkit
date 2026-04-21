import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("Please install deep-translator to run this script: pip install deep-translator")
    exit()

def load_and_translate(filepath):
    df = pd.read_csv(filepath)
    print("Original Data:")
    print(df, "\n")
    
    print("Translating reports to English...")
    translated_reports = []
    
    for idx, row in df.iterrows():
        text = row['report']
        lang = row['language']
        if lang != 'en':
            # Translate to english using deep-translator
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            translated_reports.append(translated)
        else:
            translated_reports.append(text)
            
    df['translated_report'] = translated_reports
    print("Translated Data:")
    print(df[['id', 'language', 'translated_report']], "\n")
    return df

def search_reports(df, query):
    print(f"Searching for: '{query}'\n")
    
    # Combine translated reports and the query for TF-IDF
    documents = df['translated_report'].tolist()
    documents.append(query) # Add query as the last document
    
    # Compute TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # The last row is the query vector
    query_vector = tfidf_matrix[-1]
    doc_vectors = tfidf_matrix[:-1]
    
    # Calculate cosine similarity between query and all documents
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
    # Add similarities to dataframe and sort
    df['relevance_score'] = similarities
    results = df.sort_values(by='relevance_score', ascending=False)
    
    # Filter out zero relevance
    results = results[results['relevance_score'] > 0]
    
    return results

if __name__ == "__main__":
    filepath = "reports.csv"
    try:
        df = load_and_translate(filepath)
        
        # Example query
        query = "patient has high fever and cough"
        
        results = search_reports(df, query)
        
        print(f"--- Top Search Results for '{query}' ---")
        if not results.empty:
            for idx, row in results.iterrows():
                print(f"ID: {row['id']} | Score: {row['relevance_score']:.4f} | Original: {row['report']} | English: {row['translated_report']}")
        else:
            print("No relevant reports found.")
    except Exception as e:
        print(f"Error: {e}")
