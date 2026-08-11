"""
Reproduction of:
  "Speaker-independent dysarthria severity classification using
   self-supervised transformers and multi-task learning"
  Kadirvelu et al., PLOS Digital Health, 2025
  https://doi.org/10.1371/journal.pdig.0001076

Implements:
  1. Fine-tuned wav2vec 2.0 baseline (§2.2)
  2. Speaker-Agnostic Latent Regularisation (SALR) framework (§2.3)
  3. Leave-one-subject-out cross-validation pipeline (§2.4)

Requirements:
  pip install transformers==4.33.1 torch torchaudio scikit-learn soundfile numpy
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

import os
import csv
import math
import random
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression


# ──────────────────────────────────────────────────────────────────────────────
# Constants  (all directly from the paper)
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE        = 16000   # wav2vec 2.0 native sample rate
NUM_CLASSES        = 4        # very-low / low / medium / high severity

# § 2.2 – fine-tuning hyper-parameters
BATCH_SIZE         = 4
LEARNING_RATE      = 5e-4     # 0.0005
ADAM_BETAS         = (0.9, 0.98)
ADAM_EPSILON       = 1e-8

# § 2.3 – SALR loss hyper-parameters
TRIPLET_MARGIN     = 0.05     # m  in Eq. 1
LAMBDA             = 0.01     # λ  weighting for triplet term
WARMUP_STEPS       = 3000     # α = 0 for first 3 000 steps, then α = 1

# UA-Speech severity → integer label mapping  (Table 1, § 2.1)
SEVERITY_TO_INT = {
    "very_low": 0,   # 76–100 % intelligible
    "low":      1,   # 51–75 %
    "medium":   2,   # 26–50 %
    "high":     3,   #  0–25 %
}
INT_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_INT.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Dataset
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Model Architecture
# ──────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Two-layer linear head with ReLU activation (§ 2.2).

        Linear(768 → 768) → ReLU → Linear(768 → num_classes)
    """

    def __init__(self, hidden_size: int = 768, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DysarthriaClassifier(nn.Module):
    """
    Shared backbone used by both the fine-tuned baseline and SALR.

    Architecture (§ 2.2):
        facebook/wav2vec2-base
          • 12 Transformer blocks
          • hidden size  : 768
          • FFN size     : 3 072
          • attention heads: 8
          • pretrained on 960 h LibriSpeech
        └─ mean-pool last hidden states → embedding  (B, 768)
              └─ ClassificationHead → logits  (B, 4)

    The embeddings before the classification head are also returned
    so the SALR loss can operate on them directly.
    """

    def __init__(
        self,
        model_name:  str = "facebook/wav2vec2-base",
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.wav2vec2   = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size     = self.wav2vec2.config.hidden_size          # 768
        self.classifier = ClassificationHead(hidden_size, num_classes)

    # ------------------------------------------------------------------
    # Internal helper: frame-level attention mask
    # wav2vec 2.0 downsamples audio by a factor of ≈ 320 via its CNN
    # feature extractor.  HuggingFace exposes the exact formula.
    # ------------------------------------------------------------------
    def _frame_mask(
        self,
        attention_mask: torch.Tensor,   # (B, T_audio)
        num_frames:     int,
    ) -> torch.Tensor:                  # (B, T_frames)
        """
        Project the sample-level padding mask down to frame level using
        the model's own stride calculation.
        """
        lengths = attention_mask.sum(dim=1)                         # (B,)
        frame_lengths = self.wav2vec2._get_feat_extract_output_lengths(lengths)
        mask = torch.zeros(
            attention_mask.size(0), num_frames,
            device=attention_mask.device,
        )
        for i, fl in enumerate(frame_lengths):
            fl_clamped = min(int(fl.item()), num_frames)
            mask[i, :fl_clamped] = 1.0
        return mask

    def get_embeddings(
        self,
        input_values:   torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through wav2vec 2.0 + masked mean-pool.

        Returns utterance embeddings of shape (B, 768).
        """
        out = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
        )
        hidden = out.last_hidden_state          # (B, T_frames, 768)

        if attention_mask is not None:
            fmask = self._frame_mask(attention_mask, hidden.size(1))  # (B, T_frames)
            denom = fmask.sum(dim=1, keepdim=True).clamp(min=1)
            emb   = (hidden * fmask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            emb = hidden.mean(dim=1)

        return emb                              # (B, 768)

    def forward(
        self,
        input_values:   torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits     : (B, num_classes)
        embeddings : (B, 768)   ← used by SALR triplet loss
        """
        emb    = self.get_embeddings(input_values, attention_mask)
        logits = self.classifier(emb)
        return logits, emb


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SALR Loss  (§ 2.3, Eq. 1)
# ──────────────────────────────────────────────────────────────────────────────

class SALRLoss(nn.Module):
    r"""
    Speaker-Agnostic Latent Regularisation loss.

    Combined objective (§ 2.3):

        L = α · L_CE  +  λ · L_triplet

    where
        λ = 0.01  (LAMBDA)
        α = 0     for the first 3 000 gradient steps  (warm-up phase)
        α = 1     thereafter

    Triplet construction from a single batch (Eq. 1):

        Anchor   E_AX : utterance of word A by speaker X
        Positive E_BY : utterance of word B by speaker Y ≠ X,
                        with the *same* severity as speaker X
        Negative E_BX : utterance of word B by the *same* speaker X
                        (different word from the anchor)

    Intuition: force  d(E_AX, E_BX) > d(E_AX, E_BY) + m  so that
    embeddings separate by severity rather than by speaker identity.

    Distance metric: L2 Euclidean  (p = 2 in TripletMarginLoss)
    Margin:  m = 0.05
    """

    def __init__(
        self,
        margin:       float = TRIPLET_MARGIN,
        lam:          float = LAMBDA,
        warmup_steps: int   = WARMUP_STEPS,
    ):
        super().__init__()
        self.lam          = lam
        self.warmup_steps = warmup_steps
        self.ce           = nn.CrossEntropyLoss()
        self.triplet      = nn.TripletMarginLoss(margin=margin, p=2)

    # ------------------------------------------------------------------
    def _alpha(self, step: int) -> float:
        """α schedule: 0 during warm-up, 1 afterwards."""
        return 0.0 if step < self.warmup_steps else 1.0

    # ------------------------------------------------------------------
    def _build_triplets(
        self,
        embeddings: torch.Tensor,           # (B, D)
        severities: torch.Tensor,           # (B,)
        speaker_ids: List[str],
        word_ids:    List[str],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Construct (anchor, positive, negative) triplets from the batch
        following the paper's prescription.

        For each sample i (anchor E_AX):
          negative  = random sample j from the *same* speaker, *different* word
          positive  = random sample k from a *different* speaker, *same* severity

        Returns None if the batch is too small to form any valid triplet.
        """
        B = embeddings.size(0)
        anchors, positives, negatives = [], [], []

        for i in range(B):
            sev_i = severities[i].item()
            spk_i = speaker_ids[i]
            wrd_i = word_ids[i]

            # Negative: E_BX  (same speaker, different word)
            neg_pool = [
                j for j in range(B)
                if speaker_ids[j] == spk_i and word_ids[j] != wrd_i
            ]

            # Positive: E_BY  (different speaker, same severity)
            pos_pool = [
                j for j in range(B)
                if speaker_ids[j] != spk_i and severities[j].item() == sev_i
            ]

            if not neg_pool or not pos_pool:
                continue

            neg_idx = random.choice(neg_pool)
            pos_idx = random.choice(pos_pool)

            anchors.append(embeddings[i])
            negatives.append(embeddings[neg_idx])
            positives.append(embeddings[pos_idx])

        if not anchors:
            return None, None, None

        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        logits:      torch.Tensor,
        embeddings:  torch.Tensor,
        severities:  torch.Tensor,
        speaker_ids: List[str],
        word_ids:    List[str],
        step:        int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        logits      : (B, num_classes) – output of ClassificationHead
        embeddings  : (B, 768)         – mean-pooled wav2vec2 features
        severities  : (B,)             – integer severity labels
        speaker_ids : list of length B
        word_ids    : list of length B
        step        : current global gradient step (controls α schedule)

        Returns
        -------
        loss  : scalar tensor
        info  : dict with individual loss components for logging
        """
        alpha = self._alpha(step)

        l_ce = self.ce(logits, severities)

        anch, pos, neg = self._build_triplets(
            embeddings, severities, speaker_ids, word_ids
        )
        if anch is not None:
            l_triplet = self.triplet(anch, pos, neg)
        else:
            l_triplet = embeddings.new_zeros(1).squeeze()

        loss = alpha * l_ce + self.lam * l_triplet

        return loss, {
            "loss":         loss.item(),
            "loss_ce":      l_ce.item(),
            "loss_triplet": l_triplet.item(),
            "alpha":        alpha,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    Adam as specified in § 2.2:
        lr = 0.0005,  betas = (0.9, 0.98),  eps = 1e-8
    """
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr    = LEARNING_RATE,
        betas = ADAM_BETAS,
        eps   = ADAM_EPSILON,
    )


def train_one_epoch_salr(
    model:     DysarthriaClassifier,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: SALRLoss,
    device:    torch.device,
    step:      int,
) -> Tuple[float, int]:
    """SALR multi-task training epoch. Returns (mean_loss, updated_step)."""
    model.train()
    total = 0.0

    for batch in loader:
        iv   = batch["input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        sev  = batch["severity"].to(device)

        logits, emb = model(iv, mask)
        loss, info  = criterion(
            logits, emb, sev,
            batch["speaker_id"],
            batch["word_id"],
            step,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += info["loss"]
        step  += 1

    return total / max(len(loader), 1), step


def train_one_epoch_baseline(
    model:     DysarthriaClassifier,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device:    torch.device,
) -> float:
    """Cross-entropy-only training epoch for the fine-tuned baseline."""
    model.train()
    total = 0.0

    for batch in loader:
        iv   = batch["input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        sev  = batch["severity"].to(device)

        logits, _ = model(iv, mask)
        loss      = criterion(logits, sev)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model:  DysarthriaClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Return macro accuracy and macro F1 (both ×100)."""
    model.eval()
    preds, labels = [], []

    for batch in loader:
        iv   = batch["input_values"].to(device)
        mask = batch["attention_mask"].to(device)

        logits, _ = model(iv, mask)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch["severity"].tolist())

    acc = accuracy_score(labels, preds) * 100
    f1  = f1_score(labels, preds, average="macro", zero_division=0) * 100
    return {"accuracy": acc, "f1": f1}


@torch.no_grad()
def extract_embeddings(
    model:  DysarthriaClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, severity_labels, speaker_labels) for analysis."""
    model.eval()
    embs, sevs, spks = [], [], []

    spk_to_int: Dict[str, int] = {}

    for batch in loader:
        iv   = batch["input_values"].to(device)
        mask = batch["attention_mask"].to(device)

        _, emb = model(iv, mask)
        embs.append(emb.cpu().numpy())
        sevs.extend(batch["severity"].tolist())

        for s in batch["speaker_id"]:
            if s not in spk_to_int:
                spk_to_int[s] = len(spk_to_int)
            spks.append(spk_to_int[s])

    return np.vstack(embs), np.array(sevs), np.array(spks)


def speaker_predictability(
    embeddings:  np.ndarray,
    speaker_ids: np.ndarray,
) -> float:
    """
    Train a linear classifier on embeddings to predict speaker identity (§ 3.3).
    Lower accuracy = more speaker-agnostic representations.
    """
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
    clf.fit(embeddings, speaker_ids)
    return accuracy_score(speaker_ids, clf.predict(embeddings)) * 100


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Leave-One-Subject-Out Cross-Validation  (§ 2.4)
# ──────────────────────────────────────────────────────────────────────────────

def loso_cv(
    all_samples:      List[Dict],
    dysarthric_spks:  List[str],
    processor:        Wav2Vec2Processor,
    device:           torch.device,
    model_name:       str  = "facebook/wav2vec2-base",
    num_epochs:       int  = 30,
    n_runs:           int  = 5,
    use_salr:         bool = True,
) -> Dict[str, List[float]]:
    """
    Leave-one-subject-out cross-validation pipeline (§ 2.4).

    Protocol
    --------
    • 15 iterations, one per dysarthric speaker.
    • Train  : 14 subjects × 465 utterances each
               (3 repetitions × 155 common words: digits, letters, commands,
               Brown Corpus words).
    • Test   : 300 uncommon words from the held-out subject.
    • Repeat the whole LOSO procedure n_runs = 5 times.
    • Report mean ± std of macro accuracy and macro F1.
    """
    results: Dict[str, List[float]] = {"accuracy": [], "f1": []}

    # Index samples by speaker for fast lookup
    by_speaker: Dict[str, List[Dict]] = {s: [] for s in dysarthric_spks}
    for sample in all_samples:
        spk = sample["speaker_id"]
        if spk in by_speaker:
            by_speaker[spk].append(sample)

    for run in range(n_runs):
        run_acc, run_f1 = [], []
        print(f"\n── Run {run + 1}/{n_runs} ──────────────────────────────")

        for test_spk in dysarthric_spks:
            # ── Split ─────────────────────────────────────────────────
            train_samples = [
                s for s in all_samples
                if s["speaker_id"] != test_spk and s["is_common"]
            ]
            test_samples = [
                s for s in by_speaker[test_spk]
                if not s["is_common"]           # uncommon words only
            ]

            if not test_samples:
                print(f"  Skip {test_spk}: no uncommon test samples.")
                continue

            train_ds = UASpeechDataset(train_samples, processor)
            test_ds  = UASpeechDataset(test_samples,  processor)

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                collate_fn=collate_fn, drop_last=False,
            )
            test_loader = DataLoader(
                test_ds, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn,
            )

            # ── Model ────────────────────────────────────────────────
            model     = DysarthriaClassifier(model_name).to(device)
            optimizer = build_optimizer(model)

            if use_salr:
                criterion = SALRLoss()
                step = 0
                for epoch in range(num_epochs):
                    loss, step = train_one_epoch_salr(
                        model, train_loader, optimizer, criterion, device, step
                    )
                    if (epoch + 1) % 10 == 0:
                        print(f"    epoch {epoch+1:3d}  loss={loss:.4f}  "
                              f"alpha={criterion._alpha(step):.1f}")
            else:
                criterion_base = nn.CrossEntropyLoss()
                for epoch in range(num_epochs):
                    loss = train_one_epoch_baseline(
                        model, train_loader, optimizer, criterion_base, device
                    )

            # ── Eval ─────────────────────────────────────────────────
            metrics = evaluate(model, test_loader, device)
            run_acc.append(metrics["accuracy"])
            run_f1.append(metrics["f1"])
            print(f"  Test spk {test_spk:<8s} | "
                  f"Acc {metrics['accuracy']:5.1f}%  F1 {metrics['f1']:5.1f}%")

        # ── Per-run aggregate ─────────────────────────────────────────
        mean_acc = float(np.mean(run_acc)) if run_acc else 0.0
        mean_f1  = float(np.mean(run_f1))  if run_f1  else 0.0
        results["accuracy"].append(mean_acc)
        results["f1"].append(mean_f1)
        print(f"  ── Run {run+1} mean: Acc {mean_acc:.1f}%  F1 {mean_f1:.1f}%")

    # ── Final summary (matches paper Table 2 / Table 3 format) ───────
    acc_arr = np.array(results["accuracy"])
    f1_arr  = np.array(results["f1"])
    print("\n" + "=" * 55)
    print(f"{'Model':<30} {'Accuracy':>10}  {'F1':>10}")
    print("-" * 55)
    model_label = "SALR" if use_salr else "Wav2Vec2 fine-tuned"
    print(f"{model_label:<30} "
          f"{acc_arr.mean():>6.1f}±{acc_arr.std():.1f}  "
          f"{f1_arr.mean():>6.1f}±{f1_arr.std():.1f}")
    print("=" * 55)
    # Paper benchmarks for reference:
    print("\nPaper benchmarks (Table 2, LOSO on UA-Speech):")
    print(f"  {'Wav2Vec2 fine-tuned (paper)':<35} ~54.8 %   ~43.0 %")
    print(f"  {'SALR (paper)':<35}  70.5 %    59.2 %")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Data loading  (UA-Speech)
# ──────────────────────────────────────────────────────────────────────────────

# UA-Speech word-type lists (Kim et al., Interspeech 2008).
# Common words = 155 repeated words (spoken 3× each per subject).
#   100 digits/numbers, 26 letters, 19 computer commands, 10 common words
# Uncommon words = 100 unique words per block (300 per subject, used for testing).
#
# Replace the sets below with the actual UA-Speech word IDs from the corpus.
# These are illustrative identifiers based on the dataset description.

_DIGIT_IDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
    "seventeen","eighteen","nineteen","twenty","thirty","forty","fifty",
    "sixty","seventy","eighty","ninety","hundred","thousand",
    "million","billion","point","oh","double","triple",
    "hundred_thousand","million_dollars",
    "first","second","third","fourth","fifth",
}
_LETTER_IDS = {chr(ord("a") + i) for i in range(26)}
_COMMAND_IDS = {
    "backspace","cancel","caps_lock","control","delete","end","enter",
    "escape","home","insert","page_down","page_up","shift","space",
    "tab","undo","start","stop","select",
}
COMMON_WORD_IDS = _DIGIT_IDS | _LETTER_IDS | _COMMAND_IDS  # ≈ 155 words


def intelligibility_to_severity(intelligibility: float) -> str:
    """Map % intelligibility → severity label (Table 1)."""
    if intelligibility >= 76:
        return "very_low"
    if intelligibility >= 51:
        return "low"
    if intelligibility >= 26:
        return "medium"
    return "high"


def load_ua_speech(
    data_root:     str,
    metadata_path: str,
) -> Tuple[List[Dict], List[str]]:
    """
    Load UA-Speech into a flat list of sample dicts.

    Parameters
    ----------
    data_root     : directory containing one sub-folder per speaker
                    (e.g. data_root/M01/word_id.wav)
    metadata_path : CSV with columns:
                    speaker_id, intelligibility, is_dysarthric

    Returns
    -------
    samples           : list of sample dicts
    dysarthric_spks   : list of speaker IDs with dysarthria

    Notes
    -----
    UA-Speech has 15 dysarthric speakers (M01–M12, F01–F04 approx.) and
    13 healthy controls.  Only dysarthric speakers are used in LOSO-CV.
    """
    import soundfile as sf

    meta: Dict[str, Dict] = {}
    with open(metadata_path, newline="") as fh:
        for row in csv.DictReader(fh):
            spk = row["speaker_id"].strip()
            meta[spk] = {
                "intelligibility": float(row["intelligibility"]),
                "is_dysarthric":   row["is_dysarthric"].strip().lower() == "true",
            }

    samples: List[Dict] = []
    dysarthric_spks: List[str] = []

    for spk_id, spk_meta in meta.items():
        spk_dir = os.path.join(data_root, spk_id)
        if not os.path.isdir(spk_dir):
            print(f"  Warning: directory not found for speaker {spk_id}")
            continue

        if spk_meta["is_dysarthric"]:
            dysarthric_spks.append(spk_id)

        severity_str = intelligibility_to_severity(spk_meta["intelligibility"])
        severity_int = SEVERITY_TO_INT[severity_str]

        for fname in sorted(os.listdir(spk_dir)):
            if not fname.lower().endswith(".wav"):
                continue

            word_id = os.path.splitext(fname)[0].lower()
            wav_path = os.path.join(spk_dir, fname)

            try:
                waveform, sr = sf.read(wav_path, dtype="float32")
            except Exception as e:
                print(f"  Warning: could not read {wav_path}: {e}")
                continue

            # Mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            # Resample to 16 kHz if needed
            if sr != SAMPLE_RATE:
                try:
                    import resampy
                    waveform = resampy.resample(waveform, sr, SAMPLE_RATE)
                except ImportError:
                    import torchaudio.transforms as T
                    wt = torch.from_numpy(waveform).unsqueeze(0)
                    wt = T.Resample(sr, SAMPLE_RATE)(wt)
                    waveform = wt.squeeze(0).numpy()

            # Determine common vs uncommon word
            is_common = word_id in COMMON_WORD_IDS

            samples.append({
                "waveform":   waveform,
                "speaker_id": spk_id,
                "severity":   severity_int,
                "word_id":    word_id,
                "is_common":  is_common,
            })

    print(f"Loaded {len(samples)} samples from {len(meta)} speakers "
          f"({len(dysarthric_spks)} dysarthric).")
    return samples, dysarthric_spks


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SALR – Speaker-Agnostic Latent Regularisation "
                    "(Kadirvelu et al., PLOS Digital Health 2025)"
    )
    p.add_argument("--data_root",   required=True,
                   help="Root directory of UA-Speech audio files.")
    p.add_argument("--metadata",    required=True,
                   help="Path to speaker metadata CSV "
                        "(columns: speaker_id, intelligibility, is_dysarthric).")
    p.add_argument("--model_name",  default="facebook/wav2vec2-base",
                   help="HuggingFace model identifier.")
    p.add_argument("--epochs",      type=int, default=30,
                   help="Training epochs per LOSO fold.")
    p.add_argument("--runs",        type=int, default=5,
                   help="Number of LOSO-CV repetitions (paper uses 5).")
    p.add_argument("--baseline",    action="store_true",
                   help="Run fine-tuned wav2vec2 baseline (cross-entropy only).")
    p.add_argument("--device",      default=None,
                   help="'cuda', 'mps', or 'cpu'. Auto-detected if omitted.")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device : {device}")
    print(f"Model  : {'Wav2Vec2 baseline' if args.baseline else 'SALR'}")
    print(f"Epochs : {args.epochs}  |  Runs : {args.runs}\n")

    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    samples, dysarthric_spks = load_ua_speech(args.data_root, args.metadata)

    loso_cv(
        all_samples     = samples,
        dysarthric_spks = dysarthric_spks,
        processor       = processor,
        device          = device,
        model_name      = args.model_name,
        num_epochs      = args.epochs,
        n_runs          = args.runs,
        use_salr        = not args.baseline,
    )


if __name__ == "__main__":
    main()