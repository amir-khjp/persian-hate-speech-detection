# Persian Hate Speech Detection (BERT Fine-Tuning + DistilGPT-2 Baseline) — In Progress

This repository contains notebook-based experiments for **binary Persian hate speech classification** using:
- **ParsBERT / HooshvareLab BERT (fine-tuned)** for sequence classification
- **DistilGPT-2** as an additional baseline (fine-tuned with a classification head)

> Status: **In Progress** (experiments, training settings, and reporting are being refined)

---

## Project Summary
The goal is to build and evaluate an NLP pipeline for Persian hate speech detection by:
1) Loading a Persian dataset (CSV)
2) Tokenizing with Transformer tokenizers (max length = 128)
3) Fine-tuning models with HuggingFace `Trainer`
4) Reporting test metrics (F1 + classification report)
5) Saving the best model and tokenizer for inference

---

## Dataset Format
The notebook expects these files in the same directory as the notebook:
- `train_simple.csv`
- `val_simple.csv`
- `test_simple.csv`

Expected columns (example):
- `tweet_id`
- `text`
- `HateSpeech`  *(binary label: 0/1)*  
Other columns may exist (e.g., `Violenc`, `Hate`, `Vulgar`) but this repo focuses on **HateSpeech**.

Example row:
```text
{text: "...", HateSpeech: 0/1}
