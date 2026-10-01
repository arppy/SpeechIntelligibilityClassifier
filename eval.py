import argparse
import os
import re
from typing import Dict, List, Tuple

import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor


# Import necessary modules from project files
from params import dys_speaker_dict
from dataset import (SEVERITY_TO_INT,UASpeechDataset,collate_fn)
from models import (DysarthriaClassifier)

SAMPLE_RATE = 16000
MAX_CLIP_SECONDS = 15.6


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Return macro accuracy and macro F1 (both ×100)."""
    model.eval()
    preds, labels = [], []

    for batch in loader:
        iv = batch["input_values"].to(device)
        mask = batch["attention_mask"].to(device)

        logits, _ = model(iv, mask)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch["severity"].tolist())

    acc = accuracy_score(labels, preds) * 100
    f1 = f1_score(labels, preds, average="macro", zero_division=0) * 100
    return {"accuracy": acc, "f1": f1}


def extract_speaker_from_checkpoint(checkpoint_path: str) -> str:
    """Extracts the speaker ID from the checkpoint filename (e.g., 'run4_M08_salr_ep10.pt' -> 'M08')."""
    filename = os.path.basename(checkpoint_path)
    # Matching regex pattern (e.g., M08, F01, etc.)
    match = re.search(r"_(?:run\d+_)?([A-Z]\d{2})_", "_" + filename)
    if not match:
        raise ValueError(f"Could not extract speaker ID from filename: {filename}")
    return match.group(1)


def load_speaker_uncommon_samples(
    target_speaker: str,
    data_root: str,
) -> List[Dict]:
    """
    Loads ONLY the UNCOMMON words for the specified target_speaker from the dataset.
    """

    spk_dir = os.path.join(data_root, target_speaker)
    if not os.path.isdir(spk_dir):
        raise FileNotFoundError(f"Speaker directory not found: {spk_dir}")


    severity_int = dys_speaker_dict["UASpeech"].get(target_speaker)

    wav_files = [f for f in sorted(os.listdir(spk_dir)) if f.lower().endswith(".wav")]
    print(f"[DEBUG] Found {len(wav_files)} total wav files for speaker {target_speaker}")

    samples: List[Dict] = []
    n_truncated = 0

    for fname in wav_files:
        word_id = os.path.splitext(fname)[0].lower()
        parts = word_id.split("_")

        # Extract word ID based on existing logic
        if len(parts) > 2 and parts[2].startswith("u"):
            word_id = parts[1] + "_" + parts[2]
        elif len(parts) > 2:
            word_id = parts[2]

        # KEEP ONLY UNCOMMON WORDS (starting with 'b')
        is_common = not word_id.startswith("b")
        if is_common:
            continue  # Skip common words

        wav_path = os.path.join(spk_dir, fname)

        try:
            waveform, sr = sf.read(wav_path, dtype="float32")
        except Exception as e:
            print(f"  [DEBUG Error] Could not read file {wav_path}: {e}")
            continue

        # Mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Resample to 16 kHz
        if sr != SAMPLE_RATE:
            try:
                import resampy
                waveform = resampy.resample(waveform, sr, SAMPLE_RATE)
            except ImportError:
                import torchaudio.transforms as T
                wt = torch.from_numpy(waveform).unsqueeze(0)
                wt = T.Resample(sr, SAMPLE_RATE)(wt)
                waveform = wt.squeeze(0).numpy()

        # Clip if longer than MAX_CLIP_SECONDS
        max_samples = int(MAX_CLIP_SECONDS * SAMPLE_RATE)
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]
            n_truncated += 1

        samples.append({
            "waveform": waveform,
            "speaker_id": target_speaker,
            "severity": severity_int,
            "word_id": word_id,
            "is_common": False,
        })

    print(f"[DEBUG] Loaded {len(samples)} uncommon samples for speaker {target_speaker}.")
    if n_truncated > 0:
        print(f"  Truncated clips (> {MAX_CLIP_SECONDS}s): {n_truncated}")

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dysarthria model evaluation script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="HuggingFace model identifier")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="data", help="Path to the folder where data is stored")

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Extract speaker ID from checkpoint filename
    test_spk = extract_speaker_from_checkpoint(args.checkpoint)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Selected test speaker: {test_spk}")

    # 2. Initialize Model
    model = DysarthriaClassifier(model_name=args.model_name).to(device)

    # Load Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 3. Load processor and filter samples based on extracted speaker
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    test_samples = load_speaker_uncommon_samples(test_spk, args.data_root)

    # 4. Create Dataset and DataLoader
    test_ds = UASpeechDataset(test_samples, processor)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 5. Run evaluation
    metrics = evaluate(model, test_loader, device)
    print(f"Results [{test_spk}] -> Accuracy: {metrics['accuracy']:.2f}%, F1-score: {metrics['f1']:.2f}%")