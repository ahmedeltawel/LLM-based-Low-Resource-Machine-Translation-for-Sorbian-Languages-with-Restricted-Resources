#!/usr/bin/env python3
import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from train_utils import (
    TrainConfig,
    WarmupStableDecayScheduler,
    PackedDataset,
    collate_fn,
    compute_masked_loss,
)


def save_checkpoint(model, step, config, rank):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    ckpt_dir = Path(config.output_dir) / f"step_{step:06d}"

    if rank == 0:
        for other_dir in sorted(Path(config.output_dir).glob("step_*")):
            if other_dir != ckpt_dir:
                shutil.rmtree(other_dir)
                print(f"  Deleted checkpoint: {other_dir}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    is_fsdp = isinstance(model, FSDP)
    if is_fsdp:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
    else:
        state_dict = (model.module if hasattr(model, "module") else model).state_dict()

    if rank != 0:
        return

    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(str(ckpt_dir / "model"), state_dict=state_dict)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(str(ckpt_dir / "model"))

    print(f"  Checkpoint saved: {ckpt_dir}")


def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device


def setup_logging(config, rank):
    if rank != 0:
        logging.basicConfig(level=logging.WARNING)
        return

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def main():
    ap = argparse.ArgumentParser(
        description="Continued pretraining of a Qwen3.5 model on the packed Sorbian training mix. "
                    "Launch with torchrun. Settings come from the environment: MODEL_PATH, DATA_PATH, "
                    "OUTPUT_DIR, LOG_DIR (required), PER_DEVICE_BATCH_SIZE (default 2), "
                    "GRADIENT_CHECKPOINTING (0 or 1, default 0)."
    )
    ap.parse_args()

    config = TrainConfig()

    rank, local_rank, world_size, device = setup_distributed()
    setup_logging(config, rank)

    if rank == 0:
        logging.info(f"Starting training on {world_size} GPUs")
        logging.info(f"Config: {config}")

    if rank == 0:
        logging.info(f"Loading model from {config.model_path}")

    full_cfg = AutoConfig.from_pretrained(config.model_path)
    text_cfg = full_cfg.text_config
    text_cfg.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        config=text_cfg,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.train()
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        logging.info(f"Model loaded: {param_count/1e9:.2f}B parameters")

    if world_size > 1:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
        )
        from functools import partial
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

        fsdp_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen3_5DecoderLayer},
        )

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            auto_wrap_policy=fsdp_policy,
            mixed_precision=mixed_precision,
            device_id=local_rank,
        )

        if rank == 0:
            logging.info(f"FSDP enabled: SHARD_GRAD_OP, {world_size} GPUs")

    dataset = PackedDataset(config.data_path)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    ) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        logging.info(f"Dataset: {len(dataset):,} sequences")
        logging.info(f"DataLoader: {len(dataloader):,} batches per epoch")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )

    scheduler = WarmupStableDecayScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        stable_steps=config.stable_steps,
        decay_steps=config.decay_steps,
        max_lr=config.learning_rate,
    )

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        logging.info("Starting training")
        logging.info(f"  Steps: 0 -> {config.max_steps}")
        logging.info(f"  Batch: {config.total_batch_size} (per_device={config.per_device_batch_size}, "
                     f"accum={config.grad_accum_steps}, gpus={world_size})")
        logging.info(f"  Tokens per step: {config.total_batch_size * config.max_seq_len:,}")
        logging.info(f"  LR schedule: warmup({config.warmup_steps}) -> stable({config.stable_steps}) -> decay({config.decay_steps})")

    global_step = 0
    optimizer.zero_grad()
    running_loss = 0.0
    running_tokens = 0
    step_start_time = time.time()

    epoch = 0
    while global_step < config.max_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids)
                loss, n_tokens = compute_masked_loss(outputs.logits, input_ids, loss_mask)

                scaled_loss = loss / config.grad_accum_steps

            scaled_loss.backward()

            running_loss += loss.item()
            running_tokens += n_tokens

            micro_step = (batch_idx + 1) % config.grad_accum_steps
            if micro_step == 0:
                if world_size > 1:
                    model.clip_grad_norm_(config.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                optimizer.step()
                lr = scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step % config.log_every_steps == 0:
                    elapsed = time.time() - step_start_time
                    avg_loss = running_loss / config.log_every_steps / config.grad_accum_steps
                    tokens_per_sec = running_tokens / elapsed
                    total_tokens_so_far = global_step * config.total_batch_size * config.max_seq_len

                    mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0

                    logging.info(
                        f"step {global_step:>5d}/{config.max_steps} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {lr:.2e} | "
                        f"tok/s {tokens_per_sec:,.0f} | "
                        f"tokens {total_tokens_so_far/1e9:.3f}B | "
                        f"mem {mem_gb:.1f}GB"
                    )

                    running_loss = 0.0
                    running_tokens = 0
                    step_start_time = time.time()

                if global_step % config.save_every_steps == 0:
                    if world_size > 1:
                        dist.barrier()
                    save_checkpoint(model, global_step, config, rank)
                    if world_size > 1:
                        dist.barrier()

                if global_step >= config.max_steps:
                    break

    if rank == 0:
        logging.info(f"Training complete at step {global_step}")

    if world_size > 1:
        dist.barrier()

    save_checkpoint(model, global_step, config, rank)

    if rank == 0:
        logging.info("Final checkpoint saved.")
        logging.info("Done")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
