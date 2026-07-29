#!/usr/bin/env python3
"""lerobot-train with the PaliGemma VLM frozen (pi0.5 expert-only fine-tune).

Same CLI as lerobot-train. Exists because lerobot 0.4.1 has no
train_expert_only flag for pi05, and full-parameter pi0.5 (4B params,
AdamW fp32 states ~29GB) cannot fit in an RTX 5090's 32GB VRAM.

Run on the training server (conda env `lerobot`), NOT the laptop. Example:

    python ~/train_pi05_expert_only.py \
        --dataset.repo_id=Zong-Ying/banana_towel_right_arm \
        --policy.type=pi05 \
        --policy.pretrained_path=$HOME/models/pi05_base \
        --output_dir=$HOME/outputs/pi05_banana_towel \
        --job_name=pi05_banana_towel \
        --num_workers=4 --log_freq=20 \
        --policy.compile_model=true --policy.gradient_checkpointing=true \
        --policy.dtype=bfloat16 --policy.device=cuda \
        --batch_size=8 --steps=30000 --save_freq=10000 \
        --policy.repo_id=Zong-Ying/pi05_banana_towel --policy.push_to_hub=true \
        --wandb.enable=true --wandb.project=Pi0.5_FT_BananaToTowel_MobileAloha
"""
import logging

from lerobot.policies import factory
from lerobot.scripts import lerobot_train

_orig_make_policy = factory.make_policy


def make_policy_expert_only(*args, **kwargs):
    policy = _orig_make_policy(*args, **kwargs)
    frozen = trainable = 0
    for name, p in policy.named_parameters():
        if ".paligemma_with_expert.paligemma." in name:
            p.requires_grad = False
            frozen += p.numel()
        else:
            trainable += p.numel()
    logging.info(
        f"Expert-only mode: froze VLM {frozen / 1e9:.2f}B params; "
        f"trainable {trainable / 1e6:.0f}M params"
    )
    return policy


lerobot_train.make_policy = make_policy_expert_only

if __name__ == "__main__":
    lerobot_train.main()
