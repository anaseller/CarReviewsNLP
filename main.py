# Import necessary packages
import pandas as pd
import torch
from transformers import (
    logging,
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
)
from sklearn.metrics import accuracy_score, f1_score
import evaluate

logging.set_verbosity(logging.WARNING)

# 1. Load data
df = pd.read_csv('data/car_reviews.csv', sep=';')

# Turning columns into plain lists
reviews = df['Review'].tolist()
real_labels_text = df['Class'].tolist()

# 2. Classify car reviews
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

predicted_labels = sentiment_classifier(reviews)

predictions = [1 if res['label'] == 'POSITIVE' else 0 for res in predicted_labels]
real_labels = [1 if label == 'POSITIVE' else 0 for label in real_labels_text]

# Calculate metrics
accuracy_result = accuracy_score(real_labels, predictions)
f1_result = f1_score(real_labels, predictions)

print(f"Accuracy: {accuracy_result}, F1: {f1_result}")

# 3. Translate a car review
first_review = reviews[0]
sentences = first_review.split('.')
text_to_translate = ".".join(sentences[:2]) + "."

# Explicit model & tokenizer loading for sequence-to-sequence translation
trans_model_name = "Helsinki-NLP/opus-mt-en-es"
trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)

inputs = trans_tokenizer(text_to_translate, return_tensors="pt")
outputs = trans_model.generate(**inputs)
translated_review = trans_tokenizer.decode(outputs[0], skip_special_tokens=True)

with open('data/reference_translations.txt', 'r') as f:
    reference = f.read().strip()

bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=[translated_review], references=[[reference]])

print(f"BLEU Score Dict: {bleu_score}")

# 4. Ask a question about a car review
qa_model_name = "deepset/minilm-uncased-squad2"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

context = reviews[1]  # Second review
question = "What did he like about the brand?"

inputs = qa_tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = qa_model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
answer = qa_tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

print(f"QA Answer: {answer}")

# 5. Summarize and analyze a car review
summary_model_name = "facebook/bart-large-cnn"
summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)

last_review = reviews[-1]

inputs = summary_tokenizer(last_review, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = summary_model.generate(
    inputs["input_ids"],
    min_length=50,
    max_length=55,
    no_repeat_ngram_size=2
)
summarized_text = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Toxicity and regard (с явным указанием module_type или fallback на Hub measurement)
try:
    toxicity_metric = evaluate.load("toxicity", module_type="measurement")
    toxicity_results = toxicity_metric.compute(predictions=[summarized_text])
    max_toxicity = max(toxicity_results['toxicity'])
except Exception:
    toxicity_metric = evaluate.load("evaluate-measurement/toxicity")
    toxicity_results = toxicity_metric.compute(predictions=[summarized_text])
    max_toxicity = max(toxicity_results['toxicity'])

try:
    regard_metric = evaluate.load("regard", module_type="measurement")
    regard_results = regard_metric.compute(data=[summarized_text])
except Exception:
    regard_metric = evaluate.load("evaluate-measurement/regard")
    regard_results = regard_metric.compute(data=[summarized_text])

print(f"Summary: {summarized_text}")
print(f"Max Toxicity: {max_toxicity}")

