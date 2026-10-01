import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from typing import Optional, Tuple


NUM_CLASSES = 4 # very-low / low / medium / high severity


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