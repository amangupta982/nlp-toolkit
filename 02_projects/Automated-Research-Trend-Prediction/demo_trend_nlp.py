# ============================================
# Automated Research Trend Prediction + Optimization Demo
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import torch
import torch.nn as nn

nltk.download('stopwords')
from nltk.corpus import stopwords

# ============================================
# 1. SAMPLE DATASET
# ============================================
data = {
    "year": [2019, 2020, 2021, 2022, 2023, 2024],
    "text": [
        "deep learning for image classification",
        "transformers and BERT in NLP tasks",
        "GANs for image and video generation",
        "large language models like GPT and chatbots",
        "AI in healthcare and disease prediction",
        "automation and AI in smart cities"
    ]
}

df = pd.DataFrame(data)
print("✅ Dataset Loaded:\n", df)

# ============================================
# 2. PREPROCESSING
# ============================================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean"] = df["text"].apply(clean_text)

# ============================================
# 3. FEATURE EXTRACTION (TF-IDF)
# ============================================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean"])

# ============================================
# 4. TOPIC MODELING (LDA)
# ============================================
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

terms = vectorizer.get_feature_names_out()

print("\n📌 Extracted Topics:")
for i, topic in enumerate(lda.components_):
    print(f"Topic {i}: ", [terms[i] for i in topic.argsort()[-5:]])

# ============================================
# 5. TREND ANALYSIS + PLOT
# ============================================
topic_values = lda.transform(X)

df["Topic1"] = topic_values[:, 0]
df["Topic2"] = topic_values[:, 1]

plt.figure()
plt.plot(df["year"], df["Topic1"], marker='o', label="Topic 1")
plt.plot(df["year"], df["Topic2"], marker='o', label="Topic 2")

plt.xlabel("Year")
plt.ylabel("Trend Strength")
plt.title("📈 Research Trend Prediction")
plt.legend()
plt.grid()

plt.savefig("trend_plot.png")
plt.show()

print("✅ Plot saved as trend_plot.png")

# ============================================
# 6. MODEL EVALUATION
# ============================================
perplexity = lda.perplexity(X)
print(f"\n📊 Model Perplexity: {perplexity:.2f}")

# ============================================
# 7. SIMPLE ENERGY OPTIMIZATION DEMO
# (Quantization Example)
# ============================================

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# Apply quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

print("\n⚡ Model Quantized Successfully!")

# ============================================
# 8. FINAL SUMMARY
# ============================================
print("\n🎯 Pipeline Completed:")
print("✔ Data Preprocessed")
print("✔ Topics Extracted")
print("✔ Trends Visualized")
print("✔ Model Evaluated")
print("✔ Optimization Applied")