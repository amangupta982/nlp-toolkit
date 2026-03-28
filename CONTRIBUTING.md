# Contributing to nlp-toolkit

Thanks for your interest! This repo welcomes contributions at all levels.

## What we need most

- **spaCy equivalents** of the NLTK fundamentals in `01_fundamentals/`
- **Unit tests** for preprocessing functions (pytest)
- **Better README** inside each project folder in `02_projects/`
- **Gradio demos** for any of the projects

## Getting started

```bash
git clone https://github.com/amangupta982/nlp-toolkit.git
cd nlp-toolkit
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('all')"
```

## Branch naming

```
feat/add-spacy-ner
fix/tokenizer-unicode-handling
docs/add-project-readme
```

## Commit format

```
feat: add spaCy NER implementation alongside NLTK version
fix: handle empty string edge case in text_preprocessing
docs: add usage examples to writing-style-analyzer README
test: add pytest coverage for stopword removal pipeline
```

## Good first issues

Look for the [`good first issue`](https://github.com/amangupta982/nlp-toolkit/issues?q=label%3A%22good+first+issue%22) label — these are scoped to be approachable in under 2 hours.

## Pull request checklist

- [ ] Branch created from `main`
- [ ] Code runs without errors (`python your_file.py`)
- [ ] Docstring added to any new function
- [ ] PR description explains what and why
