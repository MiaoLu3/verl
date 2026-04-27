"""SFT dataset that consumes pre-tokenized teacher rollout token tapes directly.

Bypasses verl's chat-template re-tokenization, which is essential for Qwen3
where the chat template strips ``<think>...</think>`` from non-last assistant
turns -- the resulting token tape would NOT match what the rollout model
actually saw. We therefore use the exact ``tokens.input_ids`` and
``tokens.loss_mask`` arrays the rollout produced.

Wire it into verl's SFT trainer config with::

    data.custom_cls.path=recipe/alfworld/pretokenized_sft_dataset.py
    data.custom_cls.name=PretokenizedSFTDataset

The class signature matches what ``verl.trainer.sft_trainer.create_sft_dataset``
passes (``parquet_files``, ``tokenizer``, ``config``, ``processor``,
``max_samples``). ``tokenizer`` and ``processor`` are accepted only for that
compatibility -- we use ``tokenizer.pad_token_id`` if available, otherwise 0.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch.utils.data import Dataset


class PretokenizedSFTDataset(Dataset):
    """Pre-tokenized SFT dataset.

    Args:
        parquet_files: str | list[str] -- parquet(s) produced by jsonl_to_parquet
            with --include_tokens=true, i.e. with ``tokens_input_ids`` and
            ``tokens_loss_mask`` columns (each row is a list[int]).
        tokenizer: only ``pad_token_id`` is read. Pass None to fall back to 0.
        config: dict-like with keys::
            max_length        (int, default 1024)
            truncation        ('error' | 'left' | 'right', default 'right')
            pad_mode          ('right' | 'no_padding', default 'right')
            pad_token_id      (int, default tokenizer.pad_token_id or 0)
            input_ids_key     (str, default 'tokens_input_ids')
            loss_mask_key     (str, default 'tokens_loss_mask')
            shuffle           (bool, default False)
            seed              (int, optional)
        processor: ignored.
        max_samples: int (-1 -> use all).
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: Any = None,
        config: Any = None,
        processor: Any = None,  # noqa: ARG002 -- accepted for verl compat
        max_samples: int = -1,
    ):
        cfg = config or {}
        self.max_length = int(cfg.get("max_length", 1024))
        self.truncation = cfg.get("truncation", "right")
        assert self.truncation in ("error", "left", "right"), (
            f"truncation must be one of error/left/right, got {self.truncation!r}"
        )
        self.pad_mode = cfg.get("pad_mode", "right")
        assert self.pad_mode in ("right", "no_padding"), (
            f"pad_mode must be 'right' or 'no_padding', got {self.pad_mode!r}"
        )

        cfg_pad = cfg.get("pad_token_id", None)
        if cfg_pad is not None:
            self.pad_token_id = int(cfg_pad)
        elif tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
            self.pad_token_id = int(tokenizer.pad_token_id)
        else:
            self.pad_token_id = 0

        self.input_ids_key = cfg.get("input_ids_key", "tokens_input_ids")
        self.loss_mask_key = cfg.get("loss_mask_key", "tokens_loss_mask")
        self.shuffle = bool(cfg.get("shuffle", False))
        self.seed = cfg.get("seed", None)
        self.max_samples = int(max_samples)

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        self.parquet_files = list(parquet_files)
        self._read_files()

    def _read_files(self) -> None:
        dfs = [pd.read_parquet(f, engine="pyarrow") for f in self.parquet_files]
        self.dataframe = pd.concat(dfs, ignore_index=True)

        for k in (self.input_ids_key, self.loss_mask_key):
            if k not in self.dataframe.columns:
                raise KeyError(
                    f"PretokenizedSFTDataset requires column {k!r}; got "
                    f"columns={list(self.dataframe.columns)}. Re-run "
                    f"jsonl_to_parquet.py with --include_tokens=true."
                )

        total = len(self.dataframe)
        print(f"[PretokenizedSFTDataset] dataset len: {total}")
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                idx = rng.choice(total, size=self.max_samples, replace=False)
            else:
                idx = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[idx.tolist()].reset_index(drop=True)
            print(f"[PretokenizedSFTDataset] selected {self.max_samples}/{total}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _to_long_tensor(self, arr: Any) -> torch.Tensor:
        # Parquet list columns come back as numpy arrays / lists -- both are fine.
        if isinstance(arr, np.ndarray):
            return torch.as_tensor(arr.astype(np.int64))
        return torch.as_tensor(list(arr), dtype=torch.long)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[item]
        input_ids = self._to_long_tensor(row[self.input_ids_key])
        loss_mask = self._to_long_tensor(row[self.loss_mask_key])
        if input_ids.shape != loss_mask.shape:
            raise ValueError(
                f"row {item}: tokens_input_ids and tokens_loss_mask length mismatch "
                f"({input_ids.shape} vs {loss_mask.shape})"
            )
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        seq_len = input_ids.shape[0]

        if seq_len > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"seq_len={seq_len} > max_length={self.max_length}")
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[: self.max_length]
            else:  # 'left'
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
                position_ids = position_ids[-self.max_length :]
            seq_len = input_ids.shape[0]

        if self.pad_mode == "right":
            if seq_len < self.max_length:
                pad = self.max_length - seq_len
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad,), self.pad_token_id, dtype=input_ids.dtype),
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad, dtype=attention_mask.dtype),
                ])
                loss_mask = torch.cat([
                    loss_mask,
                    torch.zeros(pad, dtype=loss_mask.dtype),
                ])
                position_ids = F.pad(position_ids, (0, pad), value=0)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        # no_padding: matches MultiTurnSFTDataset which drops attention_mask
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
