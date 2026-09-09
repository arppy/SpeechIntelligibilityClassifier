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

from collections import defaultdict
from torch.utils.data import BatchSampler
from torch.amp import autocast

# ──────────────────────────────────────────────────────────────────────────────
# Constants  (all directly from the paper)
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE        = 16000   # wav2vec 2.0 native sample rate
NUM_CLASSES        = 4        # very-low / low / medium / high severity
MAX_CLIP_SECONDS    = 15.6    # clips longer than this are truncated on load,
                               # so a single long outlier doesn't force every
                               # other sample in its batch to pad up to it

# § 2.2 – fine-tuning hyper-parameters
BATCH_SIZE         = 3
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
        self.wav2vec2   = Wav2Vec2Model.from_pretrained(model_name, gradient_checkpointing=False)
        self.wav2vec2.gradient_checkpointing_enable()
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
    Speaker-Agnostic Latent Regularisation loss (§ 2.3, Eq. 1).

        L = α · L_CE(anchor_logits, anchor_severities)
          + λ · L_triplet(anchor_emb, positive_emb, negative_emb)

    α follows a warm-up schedule: 0 for the first `warmup_steps` gradient
    steps, then 1. L_triplet uses L2 distance with margin `margin`.

    This module has no knowledge of how the anchor/negative/positive
    tensors were chosen or assembled — that happens upstream in the data
    pipeline (SALRTripletCollator) and in train_one_epoch_salr, which
    splits the model's batched outputs before calling this loss.
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

    def _alpha(self, step: int) -> float:
        """α schedule: 0 during warm-up, 1 afterwards."""
        return 0.0 if step < self.warmup_steps else 1.0

    def forward(
        self,
        anchor_logits:     torch.Tensor,  # (N, num_classes)
        anchor_severities: torch.Tensor,  # (N,)
        anchor_emb:        torch.Tensor,  # (N, D)
        positive_emb:      torch.Tensor,  # (N, D)
        negative_emb:      torch.Tensor,  # (N, D)
        step:              int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        alpha = self._alpha(step)

        l_ce      = self.ce(anchor_logits, anchor_severities)
        l_triplet = self.triplet(anchor_emb, positive_emb, negative_emb)

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
) -> Tuple[Dict[str, float], int]:
    """SALR multi-task training epoch. Returns (epoch_stats, updated_step),
    where epoch_stats has mean total/CE/triplet loss and the alpha value
    active at the end of the epoch."""
    model.train()
    totals = defaultdict(float)
    n_batches = 0

    for batch in loader:
        iv   = batch["input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        sev  = batch["severity"].to(device)
        optimizer.zero_grad()

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, emb = model(iv, mask)

            anchor_logits     = logits[0::3]
            anchor_severities = sev[0::3]
            anchor_emb        = emb[0::3]
            negative_emb      = emb[1::3]
            positive_emb      = emb[2::3]

            loss, info = criterion(
                anchor_logits, anchor_severities,
                anchor_emb, positive_emb, negative_emb,
                step,
            )

        loss.backward()
        optimizer.step()

        totals["loss"]         += info["loss"]
        totals["loss_ce"]      += info["loss_ce"]
        totals["loss_triplet"] += info["loss_triplet"]
        totals["alpha"] = info["alpha"]   # last value, not summed
        n_batches += 1
        step += 1

    epoch_stats = {
        "loss":         totals["loss"] / max(n_batches, 1),
        "loss_ce":      totals["loss_ce"] / max(n_batches, 1),
        "loss_triplet": totals["loss_triplet"] / max(n_batches, 1),
        "alpha":        totals["alpha"],
    }
    return epoch_stats, step


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
    checkpoint_dir:   str  = "./checkpoints",
    claim_ttl:        int  = 86400,
    claim_dir:        Optional[str] = None,
    base_seed:        int  = 42,  # MODOSITAS: Alap seed fogadasa
) -> Dict[str, List[float]]:
    """LOSO CV with per-fold claiming for concurrent workers."""
    results: Dict[str, List[float]] = {"accuracy": [], "f1": []}

    by_speaker: Dict[str, List[Dict]] = {s: [] for s in dysarthric_spks}
    for sample in all_samples:
        spk = sample["speaker_id"]
        if spk in by_speaker:
            by_speaker[spk].append(sample)

    tasks = [(r, s) for r in range(n_runs) for s in dysarthric_spks]

    per_run_acc = {r: [] for r in range(n_runs)}
    per_run_f1  = {r: [] for r in range(n_runs)}

    claim_storage_dir = claim_dir if claim_dir is not None else checkpoint_dir

    print(f"\nStarting LOSO with {len(tasks)} total tasks (runs x speakers)")

    while tasks:
        run, test_spk = random.choice(tasks)

        # quick check: if a checkpoint already exists for this fold — whether
        # it finished all num_epochs or not — skip it rather than resuming.
        ckpt_path = checkpoint_path(checkpoint_dir, run, test_spk, use_salr)
        if os.path.exists(ckpt_path):
            print(f"  Skip {test_spk} (run {run}): checkpoint already exists — not resuming.")
            tasks.remove((run, test_spk))
            continue

        claimed = claim_fold(claim_storage_dir, run, test_spk, use_salr, ttl=claim_ttl)
        if not claimed:
            continue

        # re-check after claiming, in case another worker created this fold's
        # checkpoint while we were in the process of claiming it.
        if os.path.exists(ckpt_path):
            release_claim(claim_storage_dir, run, test_spk, use_salr)
            tasks.remove((run, test_spk))
            continue

        # ── MÓDOSÍTÁS: Egyedi, determinisztikus seed beállítása a feladathoz ──
        spk_idx = dysarthric_spks.index(test_spk)
        task_seed = base_seed + (run * 1000) + spk_idx
        random.seed(task_seed)
        np.random.seed(task_seed)
        torch.manual_seed(task_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task_seed)
        # ─────────────────────────────────────────────────────────────────────

        train_samples = [s for s in all_samples if s['speaker_id'] != test_spk and s['is_common']]
        test_samples  = [s for s in by_speaker[test_spk] if not s['is_common']]

        if not test_samples:
            print(f"  Skip {test_spk}: no uncommon test samples.")
            release_claim(claim_storage_dir, run, test_spk, use_salr)
            tasks.remove((run, test_spk))
            continue

        train_ds = UASpeechDataset(train_samples, processor)
        test_ds  = UASpeechDataset(test_samples,  processor)

        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model     = DysarthriaClassifier(model_name).to(device)
        optimizer = build_optimizer(model)

        if use_salr:
            anchor_sampler = RandomBatchSampler(dataset_size=len(train_samples), batch_size=4)
            salr_collator  = SALRTripletCollator(train_ds)
            train_loader   = DataLoader(train_ds, batch_sampler=anchor_sampler, collate_fn=salr_collator)
            criterion      = SALRLoss()

            step         = 0
            loss_history = []

            for epoch in range(num_epochs):
                epoch_stats, step = train_one_epoch_salr(model, train_loader, optimizer, criterion, device, step)
                loss_history.append({"epoch": epoch + 1, **epoch_stats})

                print(f"    epoch {epoch+1:3d}/{num_epochs}  loss={epoch_stats['loss']:.4f}  ce={epoch_stats['loss_ce']:.4f}  triplet={epoch_stats['loss_triplet']:.4f}  alpha={epoch_stats['alpha']:.1f}")

                is_last = (epoch + 1) == num_epochs
                if (epoch + 1) % CHECKPOINT_EVERY == 0 or is_last:
                    save_checkpoint(ckpt_path, model, epoch=epoch + 1, num_epochs=num_epochs, step=step, run=run, test_spk=test_spk, use_salr=True, loss_history=loss_history)
        else:
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            criterion_base = nn.CrossEntropyLoss()

            loss_history = []

            for epoch in range(num_epochs):
                loss = train_one_epoch_baseline(model, train_loader, optimizer, criterion_base, device)
                loss_history.append({"epoch": epoch + 1, "loss": loss})
                print(f"    epoch {epoch+1:3d}/{num_epochs}  loss={loss:.4f}")

                is_last = (epoch + 1) == num_epochs
                if (epoch + 1) % CHECKPOINT_EVERY == 0 or is_last:
                    save_checkpoint(ckpt_path, model, epoch=epoch + 1, num_epochs=num_epochs, step=0, run=run, test_spk=test_spk, use_salr=False, loss_history=loss_history)

        metrics = evaluate(model, test_loader, device)
        per_run_acc[run].append(metrics['accuracy'])
        per_run_f1[run].append(metrics['f1'])
        print(f"  Test spk {test_spk:<8s} | Acc {metrics['accuracy']:5.1f}%  F1 {metrics['f1']:5.1f}%")

        release_claim(claim_storage_dir, run, test_spk, use_salr)
        tasks.remove((run, test_spk))

    for run in range(n_runs):
        run_acc = per_run_acc.get(run, [])
        run_f1  = per_run_f1.get(run, [])
        mean_acc = float(np.mean(run_acc)) if run_acc else 0.0
        mean_f1  = float(np.mean(run_f1))  if run_f1  else 0.0
        results['accuracy'].append(mean_acc)
        results['f1'].append(mean_f1)
        print(f"  ── Run {run+1} mean: Acc {mean_acc:.1f}%  F1 {mean_f1:.1f}%")

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
    n_truncated = 0

    for spk_id, spk_meta in meta.items():
        spk_dir = os.path.join(data_root, spk_id)
        if not os.path.isdir(spk_dir):
            #print(f"  Warning: directory not found for speaker {spk_id}")
            continue

        if spk_meta["is_dysarthric"]:
            dysarthric_spks.append(spk_id)

        severity_str = intelligibility_to_severity(spk_meta["intelligibility"])
        severity_int = SEVERITY_TO_INT[severity_str]

        for fname in sorted(os.listdir(spk_dir)):
            if not fname.lower().endswith(".wav"):
                continue

            word_id = os.path.splitext(fname)[0].lower()
            if word_id.split("_")[2].startswith('u'):
                word_id = word_id.split("_")[1] +"_"+ word_id.split("_")[2]
            else:
                word_id = word_id.split("_")[2]
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

            # Cut clips longer than MAX_CLIP_SECONDS (15.6 s @ 16 kHz =
            # 249 600 samples) so that padding a batch up to one long
            # outlier doesn't blow up memory.
            max_samples = int(MAX_CLIP_SECONDS * SAMPLE_RATE)
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]
                n_truncated += 1

            # Determine common vs uncommon word
            # word_id is bx_uwx for uncommon words
            if word_id.startswith("b"):
                is_common = False
            else:
                is_common = True

            samples.append({
                "waveform":   waveform,
                "speaker_id": spk_id,
                "severity":   severity_int,
                "word_id":    word_id,
                "is_common":  is_common,
            })

    print(f"Loaded {len(samples)} samples from {len(meta)} speakers "
          f"({len(dysarthric_spks)} dysarthric).")
    if n_truncated:
        print(f"  Truncated {n_truncated} clip(s) to {MAX_CLIP_SECONDS:.1f}s "
              f"({int(MAX_CLIP_SECONDS * SAMPLE_RATE)} samples).")
    return samples, dysarthric_spks

import time

# ─── Checkpointing ─────────────────────────────────────────────────────────

CHECKPOINT_EVERY = 5   # save every 5 epochs


def checkpoint_path(ckpt_dir: str, run: int, test_spk: str, use_salr: bool) -> str:
    """One file per (run, held-out speaker, model type) — the natural unit
    of parallel work: each LOSO fold is an independent model, so each
    machine can safely own a disjoint set of these files with zero
    coordination beyond "don't pick a fold someone else already claimed"."""
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = "salr" if use_salr else "baseline"
    return os.path.join(ckpt_dir, f"run{run}_{test_spk}_{tag}.pt")


def save_checkpoint(
    path:         str,
    model:        nn.Module,
    epoch:        int,
    num_epochs:   int,
    step:         int,
    run:          int,
    test_spk:     str,
    use_salr:     bool,
    loss_history: List[Dict],
) -> None:
    """Persist training state for this completed fold (used for the
    already-exists check and for post-hoc inspection of loss curves)."""
    checkpoint = {
        "model_state_dict":     model.state_dict(),
        "epoch":                epoch,
        "step":                 step,
        "run":                  run,
        "test_spk":             test_spk,
        "use_salr":             use_salr,
        "loss_history":         loss_history,
        "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Write to a temp file then atomically rename, so a crash mid-write
    # (or two machines writing at once) never leaves a half-written file
    # that another machine's existence check would pick up.
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)

    # --- MÓDOSÍTÁS A WINDOWS PERMISSIONERROR ELLEN ---
    max_retries = 5
    for attempt in range(max_retries):
        try:
            os.replace(tmp_path, path)
            break
        except PermissionError:
            if attempt == max_retries - 1:
                # Ha többszöri próbálkozásra sem sikerül, másolással írjuk felül
                import shutil
                shutil.copyfile(tmp_path, path)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            else:
                time.sleep(0.5)  # Várunk fél másodpercet, amíg a fájlzár felszabadul
    # --------------------------------------------------

    print(f"    [checkpoint] saved epoch {epoch:3d}/{num_epochs} → {path}  "
          f"(loss={loss_history[-1].get('loss', float('nan')):.4f})")


# ------------------------- Claim utilities -------------------------------
def _claim_dir_path(ckpt_dir: str, run: int, test_spk: str, use_salr: bool) -> str:
    tag = "salr" if use_salr else "baseline"
    name = f"run{run}_{test_spk}_{tag}.claim"
    return os.path.join(ckpt_dir, name)


def claim_fold(ckpt_dir: str, run: int, test_spk: str, use_salr: bool, ttl: int = 86400) -> bool:
    """Attempt to claim a fold by creating a claim directory atomically.

    Returns True if claim succeeded, False if already claimed by another worker.
    If a claim exists but is older than `ttl` seconds it will be removed and
    replaced (reclaimed).
    """
    claim_path = _claim_dir_path(ckpt_dir, run, test_spk, use_salr)
    # Use atomic file creation (O_CREAT|O_EXCL) for a reliable claim.
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(claim_path, flags)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(f"pid={os.getpid()}\n")
                fh.write(f"ts={time.time()}\n")
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
        return True
    except FileExistsError:
        # existing claim: check staleness
        try:
            mtime = os.path.getmtime(claim_path)
        except Exception:
            return False
        if time.time() - mtime > ttl:
            # stale: try to remove and claim
            try:
                os.remove(claim_path)
            except Exception:
                return False
            try:
                fd = os.open(claim_path, flags)
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                        fh.write(f"pid={os.getpid()}\n")
                        fh.write(f"ts={time.time()}\n")
                except Exception:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                return True
            except Exception:
                return False
        return False


def release_claim(ckpt_dir: str, run: int, test_spk: str, use_salr: bool) -> None:
    """Remove the claim directory if it exists. Best-effort cleanup."""
    claim_path = _claim_dir_path(ckpt_dir, run, test_spk, use_salr)
    try:
        if os.path.exists(claim_path):
            os.remove(claim_path)
    except Exception:
        pass



# ──────────────────────────────────────────────────────────────────────────────
# 7.  Custom Batch Sampling
# ──────────────────────────────────────────────────────────────────────────────


from collections import defaultdict
import random


class RandomBatchSampler(BatchSampler):
    """Yields batch_size independent random indices. No speaker/word/severity
    awareness at all — triplet resolution happens downstream, in the collate fn."""

    def __init__(self, dataset_size: int, batch_size: int = 4):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_batches = dataset_size // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            yield random.sample(range(self.dataset_size), self.batch_size)

    def __len__(self):
        return self.num_batches



# ──────────────────────────────────────────────────────────────────────────────
# 8.  Entry point
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
    p.add_argument("--claim-ttl",   type=int, default=86400,
                   help="Seconds after which an in-progress claim is considered stale.")
    p.add_argument("--claim-dir",   default=None,
                   help="Directory to store claim markers (defaults to checkpoint_dir).")
    p.add_argument("--baseline",    action="store_true",
                   help="Run fine-tuned wav2vec2 baseline (cross-entropy only).")
    p.add_argument("--device",      default=None,
                   help="'cuda', 'mps', or 'cpu'. Auto-detected if omitted.")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--checkpoint_dir", default="./checkpoints",
                   help="Directory for per-fold checkpoints (shared filesystem "
                        "if running multiple machines in parallel).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Initial Global Reproducibility
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
        all_samples=samples,
        dysarthric_spks=dysarthric_spks,
        processor=processor,
        device=device,
        model_name=args.model_name,
        num_epochs=args.epochs,
        n_runs=args.runs,
        use_salr=not args.baseline,
        checkpoint_dir=args.checkpoint_dir,
        claim_ttl=args.claim_ttl,
        claim_dir=args.claim_dir,
        base_seed=args.seed,  # MODOSITAS: Atadjuk az alapszamot
    )


if __name__ == "__main__":
    main()