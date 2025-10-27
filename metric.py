from jiwer import wer, cer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import numpy as np

# Read files
with open('original.txt', 'r', encoding='utf-8') as f:
    original = f.read().strip()

with open('created.txt', 'r', encoding='utf-8') as f:
    created = f.read().strip()

# Word Error Rate (WER)
wer_score = wer(original, created)

# Character Error Rate (CER)
cer_score = cer(original, created)

# Cosine Similarity
vectorizer = TfidfVectorizer().fit_transform([original, created])
cos_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

# Jaccard Similarity
def jaccard_similarity(a, b):
    s1, s2 = set(a.lower().split()), set(b.lower().split())
    return len(s1 & s2) / len(s1 | s2)

jaccard = jaccard_similarity(original, created)

# ROUGE Scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = scorer.score(original, created)

# Output
print("\n🎯 Text Comparison Metrics")
print("--------------------------------")
print(f"Word Error Rate (WER):        {wer_score:.4f}")
print(f"Character Error Rate (CER):   {cer_score:.4f}")
print(f"Cosine Similarity:            {cos_sim:.4f}")
print(f"Jaccard Similarity:           {jaccard:.4f}")
print(f"ROUGE-1 (Recall):             {rouge_scores['rouge1'].recall:.4f}")
print(f"ROUGE-2 (Recall):             {rouge_scores['rouge2'].recall:.4f}")
print(f"ROUGE-L (Recall):             {rouge_scores['rougeL'].recall:.4f}")