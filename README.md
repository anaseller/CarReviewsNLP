# Multi-Task NLP Pipeline for Customer Reviews (Hugging Face)

Automated NLP solution prototyping sentiment classification, translation, question answering, and review summarization using pre-trained Hugging Face transformers.

## Features & Models
- **Sentiment Analysis:** DistilBERT (Accuracy: 0.8, F1: 0.86)
- **Translation & BLEU Scoring:** Helsinki-NLP (EN -> ES) evaluated via `evaluate` library
- **Contextual QA:** MiniLM QA model for automated review insight extraction
- **Summarization & Safety:** BART Large CNN with toxicity validation (< 0.001)

## Sample Execution Output

```text
1. Sentiment Classification:
Accuracy: 1.0, F1: 1.0

2. Translation & Quality Evaluation:
BLEU Score: 0.3737

3. Contextual Question Answering:
Question: "What did he like about the brand?"
QA Answer: "its continuous commitment to durability"

4. Summarization & Safety:
Max Toxicity Score: 0.000139 (Verified Safe)