# Emotion Classification from TV Show Transcripts — Content Intelligence Agency

**Viktória Kubišová, Soheil Mohammadpour, Ron Lev Tabuchov · Breda University of Applied Sciences · 2025**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat)
![NLP](https://img.shields.io/badge/NLP-Emotion%20Classification-blueviolet?style=flat)
![CRISP-DM](https://img.shields.io/badge/Framework-CRISP--DM-blue?style=flat)

---

## Overview

This project was developed in collaboration with the Content Intelligence Agency — a company building AI-powered software to analyse the emotional content of films, series, and TV shows at scale. Working in a group of three, we built an end-to-end NLP pipeline that takes raw video as input, transcribes speech to text, translates where needed, classifies emotions per sentence, and outputs a structured file ready for downstream media analysis.

The pipeline covers every stage: speech-to-text (AssemblyAI and Whisper), feature extraction, machine translation (English → Polish → English round-trip), prompt engineering with a locally hosted LLM, and emotion classification using traditional ML, deep learning, and transformer models. The final model is a fine-tuned DeBERTa-v3-xsmall with multi-task output for primary emotion, sub-emotion, and emotional intensity.

**My individual contributions:** My primary role was analytics translator and project manager — translating technical model results into structured reports and stakeholder-facing communication, keeping the team on track, and coordinating deliverables throughout the 8-week block. On the technical side I contributed Naive Bayes and Logistic Regression model iterations and participated in prompt engineering. I wrote the full error analysis report and model card, led the final presentation, and wrote the concept and script for the group knowledge sharing podcast.

**Teammates:** Soheil Mohammadpour (LSTM, deep learning models), Ron Lev Tabuchov (RNN, BERT iterations)

---

## Research Question

> *"How can traditional NLP methods and transformer-based models be combined into an automated pipeline that accurately classifies emotions from real-world TV show transcripts, and what are the trade-offs between model complexity and performance?"*

---

## Business Context

**Client:** Content Intelligence Agency  
**Problem:** Media companies need to automatically tag emotional content in video at scale — a task that is currently manual, time-consuming, and inconsistent.

**Business Value:**
- Automated emotion tagging replaces manual annotation in content analysis workflows
- Per-sentence emotion output enables granular analysis of narrative arc and audience engagement
- Multi-task model simultaneously predicts primary emotion, sub-emotion, and intensity — richer output than single-label classifiers
- Lightweight DeBERTa-xsmall architecture can be deployed on edge devices, enabling cost-effective inference

---

## Dataset

| Property | Detail |
|----------|--------|
| Primary source | YouTube TV show transcriptions (GoEmotions, Friends, Twitter, all-groups combined) |
| Total rows | 24,000 (combined dataset); 61,018 augmented sentences added for class balance |
| Emotion classes | happiness, sadness, anger, surprise, fear, disgust, neutral (7 classes) |
| Train/test split | 90/10 |
| Test set size | 805–860 rows depending on iteration |
| Class imbalance | Neutral (49%) and happiness (30%) dominate; disgust under 1% |
| Annotation | Manual annotation of ~4,000 sentences from Dutch TV shows (Expeditie Robinson, Hunted, Big Brother); peer-reviewed against CIA pipeline labels |
| Show used | English-language show (round-translation: English → Polish → English) |

---

## Approach

| Step | Description |
|------|-------------|
| 01 Data annotation | Manually annotated TV show transcripts with 6 core Ekman emotions + neutral; peer-reviewed against CIA pipeline labels |
| 02 Speech to text | Implemented AssemblyAI and Whisper transcription pipelines; extracted sentences from a ~40-minute English-language episode |
| 03 WER calculation | Manually calculated Word Error Rate for both models; selected best transcript for downstream tasks |
| 04 Feature extraction | Extracted POS tags (NLTK), TF-IDF, sentiment scores (TextBlob + VADER), pretrained GloVe-300 embeddings, custom Word2Vec embeddings, Flesch readability, named entities, sentence/word length, exclamation counts, EmoLex lexicon features |
| 05 Model iterations | Trained and evaluated Logistic Regression *(Viktória)*, Naive Bayes *(Viktória)*, LSTM *(Soheil)*, RNN *(Ron)*, BERT and DeBERTa transformer models; 5 documented iterations with progressive improvements |
| 06 Machine translation | Fine-tuned round-translation model: English → Polish → English; manually scored 50 lines per person (3 = correct, 2 = minor error, 1 = incorrect); documented translation error patterns |
| 07 Prompt engineering | Iteratively refined prompts for locally hosted LLM emotion labelling (baseline → few-shot → definitions → structured → combined); evaluated against CIA pipeline labels |
| 08 Error analysis | Documented 5 model iterations of DeBERTa with confusion matrices, error type distributions, accuracy-by-text-length, and confidence-vs-accuracy analysis |
| 09 XAI for Transformers | Applied Gradient × Input, Layer-wise Relevance Propagation (LRP), and attention score visualisation to DeBERTa; analysed token importance and model robustness via input perturbation |
| 10 Model card | Produced full model card covering architecture, dataset, metrics, XAI, ethical considerations, and energy/sustainability analysis |
| 11 Complete pipeline | Integrated all components into a modular end-to-end pipeline: video → transcription → translation → emotion classification → structured output |
| 12 Knowledge sharing | Produced a podcast on the history of NLP as a group knowledge-sharing artifact |

**Pipeline:**

```
Video input → AssemblyAI / Whisper transcription → Round-translation (EN→PL→EN)
→ NLP feature extraction (POS, TF-IDF, VADER, GloVe, EmoLex)
→ Emotion classification (DeBERTa-v3-xsmall, multi-task)
→ Structured output: Start Time | End Time | Sentence | Translation | Emotion
```

---

## Model Results

### Best Model — Fine-tuned DeBERTa-v3-xsmall (Iteration 5)

| Metric | Score |
|--------|-------|
| Weighted F1 | **0.783** |
| Precision | 0.788 |
| Recall | 0.784 |
| Accuracy | 0.784 |
| Macro F1 | 0.67 |

**Per-class F1:**

| Emotion | F1 | Precision | Recall |
|---------|-----|-----------|--------|
| Happiness | 0.85 | 0.83 | 0.88 |
| Neutral | 0.78 | 0.79 | 0.77 |
| Surprise | 0.73 | 0.76 | 0.69 |
| Fear | 0.63 | 0.79 | 0.53 |
| Sadness | 0.61 | 0.69 | 0.55 |
| Disgust | 0.59 | 0.65 | 0.53 |
| Anger | 0.51 | 0.39 | 0.75 |

### Model Iteration Summary

| Iteration | Key change | Emotion F1 | Macro F1 |
|-----------|-----------|-----------|---------|
| 1 | Baseline — raw text only | 0.524 | — |
| 2 | + POS, TF-IDF, VADER, TextBlob features | 0.527 | — |
| 3 | + EmoLex features | 0.507 | — |
| 4 | Hierarchical sub-emotion → emotion mapping (no retraining) | **0.801** | 0.18 |
| **5** | **+ Text augmentation + TF-IDF + EmoLex on raw test set** | **0.783** | **0.67** |

Iteration 4 produced the highest weighted F1 (0.801) through hierarchical post-processing alone, but at the cost of minority class recall. Iteration 5 sacrificed a small amount of top-line F1 to achieve a far more balanced result — macro F1 rose from 0.18 to 0.67, and anger, previously undetectable, reached F1 0.51.

### Traditional Models *(my contributions)*

Logistic Regression and Naive Bayes were trained across multiple iterations using different preprocessing levels (raw, light, medium, heavy) and feature combinations (TF-IDF, GloVe embeddings, NLP features). Both served as interpretable baselines and informed feature selection for the deep learning models.

---

## Error Analysis

Five documented iterations revealed consistent patterns: neutral-to-happiness confusion was the dominant error throughout, accounting for up to 75% of all mistakes. Short, ambiguous texts and positive-toned neutral statements were the most frequent failure points. Key findings:

- Iteration 1 baseline: neutral-to-happiness confusion accounted for 275 instances (confidence 0.552)
- Feature extraction (Iteration 2) reduced this by 28% but did not improve overall accuracy
- EmoLex features (Iteration 3) increased model confidence in wrong predictions — overconfident misclassification
- Hierarchical sub-emotion mapping (Iteration 4) was the most impactful single change, boosting accuracy from 0.38 to 0.734
- Data augmentation (Iteration 5) resolved minority class invisibility; macro F1 jumped from 0.18 to 0.67

Full report: `task 8 - Error analysis/Error_Analysis.pdf`

---

## Explainable AI

XAI was applied to the final DeBERTa model using three methods — Gradient × Input, Conservative Propagation (LRP), and attention score visualisation — across 18 sentences (3 per emotion class). Key findings: the two methods often showed low agreement, capturing different aspects of model reasoning; emotion-specific attention patterns emerged (anger emphasised emotional words; fear focused on threat-related terms); model confidence was non-linear with token removal, indicating complex token interdependencies.

Full report: `task 9 - XAI for Transformers/DeBERTa/Explainable AI (XAI) Report.md`

---

## Assumptions & Limitations

- English-language show used; round-translation (EN→PL→EN) introduced translation noise documented in `translations_scored.xlsx`
- Class imbalance (neutral 49%, disgust <1%) persists despite augmentation; minority class performance remains lower
- Model trained and evaluated on TV show transcripts — generalisation to other domains (news, social media) is unvalidated
- Prompt engineering evaluated against CIA pipeline labels rather than human ground truth
- Inference energy estimated at 20–40 W; training consumed ~0.05 kWh per session on RTX 3060

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python (PyTorch, TensorFlow) | Model training |
| HuggingFace Transformers | DeBERTa, BERT fine-tuning |
| scikit-learn | Logistic Regression, Naive Bayes, TF-IDF |
| NLTK, SpaCy, TextBlob, VADER | Feature extraction and NLP preprocessing |
| GloVe, Word2Vec (Gensim) | Word embeddings |
| AssemblyAI, Whisper | Speech-to-text transcription |
| NRC EmoLex | Emotion lexicon features |
| Locally hosted LLM | Prompt engineering |
| OPUS / HuggingFace datasets | Machine translation training data |

---

## Deliverables

- `task 1 - Annotation/` — Annotated emotion dataset and SWOT analysis + team agreements
- `task 2 - Speech to Text/` — AssemblyAI and Whisper transcription scripts and output files
- `task 3 - WER rate Calculation/` — WER annotation files for both models
- `task 4 - Feature Extraction/` — NLP feature extraction notebook (`NLP_features.ipynb`) and output (`NLP_features.xlsx`)
- `task 5 - Model Iterations/` — Logistic Regression, Naive Bayes, LSTM, RNN, BERT, and DeBERTa notebooks with saved model weights; `Y2C_model_iteration_log.xlsx`
- `task 6 - Machine Translation/` — Round-translation notebooks (EN→PL, PL→EN), scored translations, and findings report
- `Task 7 - Prompt/` — Prompt engineering scripts (baseline, few-shot, definitions, structured, combined) and evaluation
- `task 8 - Error analysis/` — `Error_Analysis.pdf` — 5-iteration DeBERTa error analysis report
- `task 9 - XAI for Transformers/` — XAI notebooks and reports for DeBERTa and LSTM
- `task 10 - model card/` — `Model_card.md` — full model card
- `task 11 - Complete Pipeline/` — Modular pipeline (`src/pipeline.py`, `src/emotion_classifier.py`, `src/speech_to_text.py`, `src/translator.py`), saved models, `README.md`, `requirements.txt`
- `task 12 - Final Presentation/` — `NLP - BUas x CIA - Group 21.pdf`
- `task 14 - Knowledge Sharing/` — `task14_podcast_compressed.mp3` — podcast on the history of NLP

---

*Developed for the Content Intelligence Agency as part of the Year 2 Block C Natural Language Processing module, BSc Applied Data Science & AI programme at Breda University of Applied Sciences.*
