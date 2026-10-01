import torch
import random

from sympy.strategies.core import switch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

SAMPLE_RATE        = 16000   # wav2vec 2.0 native sample rate

# UA-Speech severity → integer label mapping  (Table 1, § 2.1)
SEVERITY_TO_INT = {
    "very_low": 0,   # 76–100 % intelligible
    "low":      1,   # 51–75 %
    "medium":   2,   # 26–50 %
    "high":     3,   #  0–25 %
}

def label_to_severity(label: float) -> str:
    """Map % intelligibility → severity label (Table 1)."""
    match label:
        case 0:
            return "very_low"
        case 1:
            return "low"
        case 2:
            return "medium"
        case 3:
            return "high"
        case _:
            raise ValueError(f"Invalid severity label: {label}")


class UASpeechDataset(Dataset):
    """
    Wraps pre-loaded sample dicts for use with a DataLoader.

    Each sample dict must contain:
        waveform      (np.ndarray, float32, 16 kHz, mono)
        speaker_id    (str)
        severity      (int  0–3)
        word_id       (str, unique word identifier)
        is_common     (bool, True for the 155 repeated common words)
    """

    def __init__(
        self,
        samples:     List[Dict],
        processor:   Wav2Vec2Processor,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.samples     = samples
        self.processor   = processor
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        proc = self.processor(
            item["waveform"],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        return {
            "input_values": proc.input_values.squeeze(0),   # (T,)
            "severity":     item["severity"],
            "speaker_id":   item["speaker_id"],
            "word_id":      item["word_id"],
            "idx": idx,  # NEW — lets SALRTripletCollator find this sample again
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Pad variable-length waveforms to the longest in the batch."""
    max_len = max(b["input_values"].shape[0] for b in batch)

    padded_iv  = torch.zeros(len(batch), max_len)
    attn_mask  = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_values"].shape[0]
        padded_iv[i, :L]  = b["input_values"]
        attn_mask[i, :L]  = 1

    return {
        "input_values": padded_iv,
        "attention_mask": attn_mask,
        "severity":      torch.tensor([b["severity"]    for b in batch], dtype=torch.long),
        "speaker_id":    [b["speaker_id"] for b in batch],
        "word_id":       [b["word_id"]    for b in batch],
    }
class SALRTripletCollator:
    """
    Turns a batch of raw anchor indices into a full (Anchor, Negative,
    Positive) triplet batch: for each anchor, finds a same-speaker,
    different-word "negative" and a different-speaker (same severity),
    same-word-as-negative "positive", then fetches and processes their
    waveforms. If an anchor can't form a triplet, draws a fresh
    replacement and retries.

    Returned batches are laid out [A, N, P, A, N, P, ...] — this ordering
    convention is what train_one_epoch_salr splits back out before
    calling SALRLoss.
    """

    def __init__(self, dataset: "UASpeechDataset", max_resample_attempts: int = 50):
        self.dataset = dataset
        self.tree = self._build_tree(dataset.samples)
        self.max_resample_attempts = max_resample_attempts

    @staticmethod
    def _build_tree(dataset_samples: List[Dict]) -> Dict:
        """severity -> speaker -> word -> [indices]."""
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for idx, sample in enumerate(dataset_samples):
            tree[sample["severity"]][sample["speaker_id"]][sample["word_id"]].append(idx)
        return tree

    def _find_triplet(self, anchor_idx: int) -> Optional[Tuple[int, int, int]]:
        """Resolve one anchor into (anchor_idx, negative_idx, positive_idx), or None."""
        anchor = self.dataset.samples[anchor_idx]
        sev, spk_x, word_a = anchor["severity"], anchor["speaker_id"], anchor["word_id"]

        negative_candidates = []
        for word_b, indices in self.tree[sev][spk_x].items():
            if word_b != word_a:
                negative_candidates.extend((word_b, i) for i in indices)
        random.shuffle(negative_candidates)

        for word_b, negative_idx in negative_candidates:
            positive_candidates = []
            for spk_y, word_dict in self.tree[sev].items():
                if spk_y != spk_x and word_b in word_dict:
                    positive_candidates.extend(word_dict[word_b])
            if positive_candidates:
                return anchor_idx, negative_idx, random.choice(positive_candidates)

        return None

    def _resolve(self, anchor_idx: int) -> Tuple[int, int, int]:
        for _ in range(self.max_resample_attempts):
            triplet = self._find_triplet(anchor_idx)
            if triplet is not None:
                return triplet
            anchor_idx = random.randrange(len(self.dataset.samples))
        raise RuntimeError(f"Could not resolve a triplet after {self.max_resample_attempts} attempts.")

    def __call__(self, batch_items: List[Dict]) -> Dict:
        resolved_indices = []
        for item in batch_items:
            resolved_indices.extend(self._resolve(item["idx"]))
        items = [self.dataset[i] for i in resolved_indices]
        return collate_fn(items)