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



**Note:** Dataset files are **not included** in this repository.

---

## Hardware

Training was tested on:

* **GPU:** NVIDIA GeForce **MX150**
* Mixed precision (`fp16=True`) + gradient accumulation were used to fit GPU memory.

---

## Results (Current)

### 1) Fine-tuned ParsBERT (HooshvareLab/bert-fa-zwnj-base)

* **Validation F1:** ~0.75 (best epoch around 2)
* **Test F1:** **~0.753**
* **Test accuracy:** **~0.78**

Classification report (test):

* Normal: Precision ~0.80, Recall ~0.82, F1 ~0.81
* HateSpeech: Precision ~0.77, Recall ~0.74, F1 ~0.75

### 2) Raw (Untrained) ParsBERT Head (Expected weak)

As expected, using the base model with a randomly initialized classification head produces low performance.

### 3) Fine-tuned DistilGPT-2 Baseline

* **Test accuracy:** ~0.58
* Performance is currently lower than ParsBERT for this task (expected due to model mismatch + training constraints).

---

## Repository Contents

* `main-bert-gpu.ipynb` — main experiments notebook
* `.gitignore` — prevents pushing datasets, checkpoints, and large model files
* `README.md`

---

## Setup

### 1) Install dependencies

Create a fresh environment (recommended), then install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets evaluate scikit-learn tqdm
```

If you prefer:

```bash
pip install -r requirements.txt
```

### 2) Run the notebook

Open Jupyter and run:

```bash
jupyter notebook
```

Then open:

* `main-bert-gpu.ipynb`

---

## Training Notes (MX150-friendly)

Recommended settings used in experiments:

* `fp16=True`
* small batch size (e.g., 4)
* `gradient_accumulation_steps` to simulate larger effective batch
* `gradient_checkpointing=True` (optional)

---

## Saving the Model

After training, save the best model + tokenizer:

```python
trainer.save_model("./parsbert-hate-speech-model")
tokenizer.save_pretrained("./parsbert-hate-speech-model")
```

---

## Common Issues (Windows)

### HuggingFace symlink warning

On Windows, HuggingFace may warn about symlinks. It’s usually safe to ignore.
To remove the warning, enable **Developer Mode** or run as Administrator.

### Download timeouts (model.safetensors)

If downloads time out, re-run the cell; HuggingFace will often resume.
(Installing `hf_xet` may improve performance, but it’s optional.)

---

## Git / Large Files Warning

Do **not** push training outputs like:

* `checkpoint-*`
* `*.safetensors`, `*.pth`, `*.bin`

If you need to publish trained weights, use:

* GitHub Releases, or
* HuggingFace Hub, or
* Git LFS (only if necessary)

---

## License

This project is currently shared for research/learning purposes.
(Add a license file if you plan to make it open-source.)

```

If you want, I can also generate a **.gitignore** tailored to your exact folders (so checkpoints never get committed again), and a short “About” + Topics list for the repo.
```

