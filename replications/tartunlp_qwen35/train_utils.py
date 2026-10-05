import os
import torch
from dataclasses import dataclass, field
from datasets import load_from_disk


def required_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


@dataclass
class TrainConfig:
    model_path: str = field(default_factory=lambda: required_env("MODEL_PATH"))
    data_path: str = field(default_factory=lambda: required_env("DATA_PATH"))
    output_dir: str = field(default_factory=lambda: required_env("OUTPUT_DIR"))
    log_dir: str = field(default_factory=lambda: required_env("LOG_DIR"))

    learning_rate: float = 1e-4
    adam_eps: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    max_seq_len: int = 4096
    total_batch_size: int = 128
    max_steps: int = 2391
    warmup_steps: int = 256
    decay_steps: int = 768
    precision: str = "bfloat16"

    fsdp_strategy: str = "SHARD_GRAD_OP"

    per_device_batch_size: int = field(default_factory=lambda: int(os.environ.get("PER_DEVICE_BATCH_SIZE", "2")))
    gradient_checkpointing: bool = field(default_factory=lambda: bool(int(os.environ.get("GRADIENT_CHECKPOINTING", "0"))))
    save_every_steps: int = 200
    log_every_steps: int = 10
    eval_every_steps: int = 500
    grad_clip: float = 1.0

    @property
    def stable_steps(self):
        return self.max_steps - self.warmup_steps - self.decay_steps

    @property
    def grad_accum_steps(self):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        per_step = self.per_device_batch_size * world_size
        accum = self.total_batch_size // per_step
        if accum * per_step != self.total_batch_size:
            raise ValueError(
                f"total_batch_size ({self.total_batch_size}) must be divisible by "
                f"per_device_batch_size ({self.per_device_batch_size}) * world_size ({world_size}). "
                f"Got {accum * per_step}. "
                f"Try per_device_batch_size in: {[b for b in [1,2,4,8,16,32] if self.total_batch_size % (b * world_size) == 0]}"
            )
        return accum


class WarmupStableDecayScheduler:
    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.max_lr = max_lr
        self.stable_end = warmup_steps + stable_steps
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self._step = 0

    def get_lr(self):
        if self._step < self.warmup_steps:
            return self.max_lr * self._step / max(self.warmup_steps, 1)
        elif self._step < self.stable_end:
            return self.max_lr
        else:
            decay_progress = (self._step - self.stable_end) / max(self.decay_steps, 1)
            return self.max_lr * max(1.0 - decay_progress, 0.0)

    def step(self):
        self._step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.ds = load_from_disk(data_path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "loss_mask": torch.tensor(row["loss_mask"], dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "loss_mask": torch.stack([b["loss_mask"] for b in batch]),
    }


def compute_masked_loss(logits, input_ids, loss_mask):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()

    B, T, V = shift_logits.shape
    flat_logits = shift_logits.view(B * T, V)
    flat_labels = shift_labels.view(B * T)
    flat_mask = shift_mask.view(B * T)

    loss_per_token = torch.nn.functional.cross_entropy(
        flat_logits, flat_labels, reduction="none"
    )

    masked_loss = loss_per_token * flat_mask
    num_loss_tokens = flat_mask.sum()

    if num_loss_tokens > 0:
        loss = masked_loss.sum() / num_loss_tokens
    else:
        loss = masked_loss.sum()

    return loss, num_loss_tokens.item()
