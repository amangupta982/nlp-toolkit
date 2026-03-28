# 🧠 NLP Toolkit

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green?logo=python)](https://nltk.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.x-09A3D5?logo=spacy)](https://spacy.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A structured collection of NLP fundamentals and end-to-end projects — from tokenization and POS tagging to multi-model text classification and graph-based NLP systems.

Built to develop deep, practical understanding of Natural Language Processing using NLTK and spaCy — going beyond tutorials to real, working implementations.

---

## 📁 Repository Structure

```
nlp-toolkit/
├── 01_fundamentals/           # Core NLP building blocks
│   ├── text_preprocessing.py      # Cleaning, normalization, stemming
│   ├── tokenization_chatbot.py    # Rule-based chatbot with tokenization
│   ├── pos_tagging.py             # Part-of-speech tagging with NLTK
│   ├── remove_stopwords.py        # Stopword filtering pipeline
│   ├── nltk_chunking_parser.py    # Chunking and shallow parsing
│   ├── named_entity_recognition.py    # NER with NLTK
│   └── simple_chatbot.py          # Pattern-matching chatbot
│
├── 02_projects/               # End-to-end NLP applications
│   ├── cricket-commentary-generation/   # NLP-based sports commentary generator
│   ├── graph-builder-nlp/               # Graph construction from text using NLP
│   ├── multi-model-project/             # Comparing multiple NLP models on same task
│   ├── multi-task-logistic-regression/  # Multi-class text classification
│   └── writing-style-analyzer/         # Authorship and style analysis tool
│
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## 🔧 01 — Fundamentals

Core NLP building blocks implemented from scratch using NLTK and spaCy. Each file is self-contained and runnable independently.

| File | What it demonstrates |
|------|----------------------|
| `text_preprocessing.py` | Lowercasing, punctuation removal, stemming, lemmatization |
| `tokenization_chatbot.py` | Word and sentence tokenization driving a rule-based chatbot |
| `pos_tagging.py` | Penn Treebank POS tags, chunking grammar rules |
| `remove_stopwords.py` | NLTK stopword corpus, custom stopword extension |
| `nltk_chunking_parser.py` | Regex-based chunking, noun phrase extraction |
| `named_entity_recognition.py` | NLTK NE chunker — PERSON, ORG, GPE detection |
| `simple_chatbot.py` | Pattern matching, intent detection without ML |

### Quick Start

```bash
git clone https://github.com/amangupta982/nlp-toolkit.git
cd nlp-toolkit
pip install -r requirements.txt
python 01_fundamentals/text_preprocessing.py
```

---

## 🚀 02 — Projects

### 🏏 Cricket Commentary Generation
Generates cricket match commentary from structured match data using NLP templates and text generation techniques.
```bash
cd 02_projects/cricket-commentary-generation
python main.py
```

### 🕸️ Graph Builder NLP
Constructs knowledge graphs from raw text — extracts entities and relationships to build a queryable graph structure.
```bash
cd 02_projects/graph-builder-nlp
python graph_builder.py
```

### 🔀 Multi-Model Project
Benchmarks multiple NLP approaches (bag-of-words, TF-IDF, embeddings) on the same text classification task. Includes performance comparison table.
```bash
cd 02_projects/multi-model-project
python compare_models.py
```

### 📊 Multi-Task Logistic Regression
Multi-class text classifier using logistic regression with TF-IDF features. Supports multiple simultaneous classification tasks.
```bash
cd 02_projects/multi-task-logistic-regression
python train.py
```

### ✍️ Writing Style Analyzer
Analyzes authorship signals — sentence length distribution, vocabulary richness, POS patterns — to distinguish writing styles.
```bash
cd 02_projects/writing-style-analyzer
python analyze.py
```

---

## ⚙️ Setup

```bash
# Clone
git clone https://github.com/amangupta982/nlp-toolkit.git
cd nlp-toolkit

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first time only)
python -c "
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
"
```

---

## 🗺️ Learning Path

If you're using this repo to learn NLP, follow this order:

```
1. text_preprocessing.py      → Start here: the foundation of all NLP
2. remove_stopwords.py        → Filtering noise
3. tokenization_chatbot.py    → Tokenization in a real use case
4. pos_tagging.py             → Understanding sentence grammar
5. nltk_chunking_parser.py    → Extracting meaningful phrases
6. named_entity_recognition.py → Finding real-world entities
7. simple_chatbot.py          → Putting it together
        ↓
8. writing-style-analyzer     → First real project
9. multi-task-logistic-regression → ML meets NLP
10. multi-model-project       → Compare approaches
11. graph-builder-nlp         → Advanced: structured knowledge
12. cricket-commentary-generation → Creative NLP application
```

---

## 🔭 Roadmap

- [ ] Add spaCy equivalents for each NLTK fundamental
- [ ] Add unit tests for all preprocessing functions
- [ ] Add HuggingFace Transformers comparison notebook
- [ ] Deploy writing-style-analyzer as a Gradio demo
- [ ] Add benchmark results table for multi-model-project

> See [`good first issue`](https://github.com/amangupta982/nlp-toolkit/issues?q=label%3A%22good+first+issue%22) for contributor-friendly tasks.

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All PRs welcome — especially spaCy implementations of the NLTK fundamentals.

---

## 📄 License

[MIT](LICENSE) © 2025 Aman Gupta · [@amangupta982](https://github.com/amangupta982)