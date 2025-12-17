# train.py
"""
Train an English RoBERTa/BERT-style classifier on bert_model/dataset.csv.
Saves model + tokenizer + label_encoder.pkl to bert_model/saved_model

Usage:
    cd bert_model
    python train.py
Optionally set ROBERTA_MODEL env var to a different HF model, e.g.:
    ROBERTA_MODEL=roberta-large python train.py
"""
# >>> Put this at the very top of train.py BEFORE other HF imports
import os
os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_cache\transformers"
os.environ["HF_DATASETS_CACHE"] = r"E:\hf_cache\data"
os.environ["TMP"] = r"E:\hf_cache\tmp"
os.environ["TEMP"] = r"E:\hf_cache\tmp"
# <<< end of HF cache overrides

import random
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")   # <- expects dataset.csv inside bert_model/
OUTPUT_DIR = os.path.join(BASE_DIR, "saved_model")
ROBERTA_MODEL = os.environ.get("ROBERTA_MODEL", "roberta-base")  # change if you have resources
SEED = 42

# CPU-friendly defaults
BATCH_SIZE = 4         # adjust upward if you have more RAM/cores
NUM_EPOCHS = 6
LR = 2e-5
MAX_LEN = 128          # short for faster CPU training; increase if paragraphs are long
TEST_SIZE = 0.20
VAL_SIZE = 0.10
FP16 = False           # keep False for CPU

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- utilities ----------------
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all()

# ---------------- load data ----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"dataset.csv not found at expected path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
# choose text column priority: prefer text_translated -> text -> text_orig
for col in ("text_translated", "text", "text_orig"):
    if col in df.columns:
        text_col = col
        break
else:
    raise RuntimeError("No text column found in dataset CSV. Expected one of: text_translated, text, text_orig.")

print("Using text column:", text_col)

df[text_col] = df[text_col].fillna("").astype(str)
if "label" not in df.columns:
    raise RuntimeError("dataset.csv must contain a 'label' column with department names.")
df = df[[text_col, "label"]].dropna()
df = df.rename(columns={text_col: "text"})
# drop empty texts
df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
print("Total examples (after cleaning):", len(df))

# ---------------- labels ----------------
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])
num_labels = len(le.classes_)
print("Labels:", list(le.classes_), "num:", num_labels)

# save label encoder early (will be overwritten at end with same object)
with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# ---------------- split ----------------
train_val, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["label_id"], random_state=SEED)
train, val = train_test_split(train_val, test_size=VAL_SIZE, stratify=train_val["label_id"], random_state=SEED)
print("Split sizes -> train:", len(train), "val:", len(val), "test:", len(test))

train_ds = Dataset.from_pandas(train[["text", "label_id"]])
val_ds = Dataset.from_pandas(val[["text", "label_id"]])
test_ds = Dataset.from_pandas(test[["text", "label_id"]])
dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

# ---------------- tokenizer & model ----------------
print("Loading model:", ROBERTA_MODEL)
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL, use_fast=True)

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding=False, max_length=MAX_LEN)

dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
dataset = dataset.rename_column("label_id", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL, num_labels=num_labels)

# ---------------- data collator ----------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------- metrics ----------------
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    p_, r_, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": p_, "recall": r_, "f1": f1}

# ---------------- trainer ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=3,
    fp16=FP16,
    seed=SEED,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

# ---------------- train ----------------
print("Starting training...")
trainer.train()

# ---------------- evaluate ----------------
print("Evaluating on test set...")
metrics = trainer.evaluate(dataset["test"])
print("Test metrics:", metrics)

pred_out = trainer.predict(dataset["test"])
y_true = pred_out.label_ids
y_pred = np.argmax(pred_out.predictions, axis=1)

print("\nClassification report (per-class):\n")
print(classification_report(y_true, y_pred, target_names=list(le.classes_), zero_division=0))
print("\nConfusion matrix:\n")
print(confusion_matrix(y_true, y_pred))

# ---------------- save ----------------
print("Saving model and tokenizer...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
print("Saved model to:", OUTPUT_DIR)
