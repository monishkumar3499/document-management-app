"""
DepartmentClassifier: loads saved_model from bert_model/saved_model and exposes
 - classify_text(text): single paragraph
 - classify_batch(texts): batched model-only predictions
By default this version forces CPU-only inference for safety in CPU environments.
"""
import os
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
DEFAULT_MAX_LEN = 512


class DepartmentClassifier:
    def __init__(self, model_dir=MODEL_DIR, device: str = None):
        # Force CPU by default. If you want GPU later, pass device="cuda" or:
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        # load tokenizer & model (Auto* will pick Roberta* based on config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

        # load label encoder
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder: LabelEncoder = pickle.load(f)
        self.labels = list(self.label_encoder.classes_)

    def _tokenize_batch(self, texts: List[str], max_length: int = DEFAULT_MAX_LEN):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def classify_batch(self, texts: List[str], top_k: int = 2) -> List[Dict]:
        """
        Batch-classify a list of paragraph texts using ONLY the trained model.

        Returns list of dicts:
          - top_model_preds: list[(label, prob), ...] (up to top_k)
          - model_scores: dict[label -> prob] for all labels
          - final_department: chosen department (argmax)
        """
        if not texts:
            return []

        enc = self._tokenize_batch(texts, max_length=DEFAULT_MAX_LEN)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)
            probs = F.softmax(out.logits, dim=-1).cpu()  # (batch, num_labels)

        results = []

        for i, _text in enumerate(texts):
            prob_vec = probs[i]  # shape: (num_labels,)

            # top-k predictions (by model prob only)
            k = min(top_k, prob_vec.size(0))
            topk = torch.topk(prob_vec, k=k)
            topk_idxs = topk.indices.tolist()
            topk_probs = topk.values.tolist()
            topk_labels = self.label_encoder.inverse_transform(topk_idxs)

            final_department = topk_labels[0]

            # full distribution as dict: label -> prob
            model_scores = {
                label: float(prob_vec[self.label_encoder.transform([label])[0]])
                for label in self.labels
            }

            results.append({
                "top_model_preds": list(zip(topk_labels, topk_probs)),
                "model_scores": model_scores,
                "final_department": final_department
            })

        return results

    def classify_text(self, text: str, **kwargs):
        """
        Convenience wrapper for single-text classification.
        """
        return self.classify_batch([text], **kwargs)[0]


# Quick local test when running file directly
if __name__ == "__main__":
    clf = DepartmentClassifier()
    samples = [
        "Employee leave policy update and payroll changes.",
        "Tunnel ventilation system performance issue detected on platform 3."
    ]
    out = clf.classify_batch(samples)
    for s, r in zip(samples, out):
        print("TEXT:", s)
        print("PRED:", r)
        print("---")
