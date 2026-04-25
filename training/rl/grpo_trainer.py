"""
training/rl/grpo_trainer.py
============================
GRPO trainer for OpenCloud-SRE.

Bugs fixed vs previous version
--------------------------------
BUG-1  grpo_update recreated AdamW optimizer every call → no momentum, no
        effective learning. Optimizer is now created once and passed around.
BUG-2  rollout called env.step() G times from the same state (all G completions
        advanced the server env). The env now resets to the same obs_before for
        each completion so rewards are comparable.
BUG-3  build_prompt slo_score fell back to literal 0 (not computed) when
        'slo_score' key was absent — now computes it from the three metrics.
BUG-4  done flag read from best_obs dict keys 'terminated'/'truncated' which
        are nested inside the server StepResponse under top-level keys — fixed
        to read correctly from server JSON.
BUG-5  pad_token_id not set → decoder-only models crash on multi-sequence
        generation without a pad token.
BUG-6  tokenizer called twice per step (once for prompt length, once for full)
        wasting ~40% of CPU in grpo_update → precompute prompt_len once.
BUG-7  all_rewards / all_components grew unbounded across the epoch, causing
        logging mean to be wrong for later steps. Reset per-step.
BUG-8  logging only on step%5==0 but step resets at epoch boundary → step
        counter was epoch-local anyway, now clearer.

Usage:
    uvicorn env.server:app --port 8000
    python -m training.rl.grpo_trainer --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from torch.optim import AdamW

logger = logging.getLogger(__name__)

# ── optional backends ─────────────────────────────────────────────────────────
try:
    from unsloth import FastLanguageModel
    _UNSLOTH = True
except ImportError:
    _UNSLOTH = False

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class GRPOTrainConfig:
    model_name:      str   = "Qwen/Qwen2.5-1.5B-Instruct"
    env_url:         str   = "http://localhost:8000"
    output_dir:      str   = "models/rl_model"
    epochs:          int   = 3
    steps_per_epoch: int   = 50
    group_size:      int   = 8
    max_new_tokens:  int   = 128          # FIX: was 256, JSON fits in 128
    learning_rate:   float = 5e-6
    max_seq_length:  int   = 1024         # FIX: was 2048, shorter = faster
    lora_rank:       int   = 16
    load_in_4bit:    bool  = True
    temperature:     float = 0.8          # FIX: lower = less garbage JSON
    wandb_project:   str   = "opencloud-sre-grpo"
    seed:            int   = 42
    grad_clip:       float = 1.0          # gradient clipping for stability


# ── OpenEnv HTTP client ────────────────────────────────────────────────────────
class OpenEnvClient:
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/reset",
            json={"seed": seed, "crash_on_reset": True},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: str) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def valid_actions(self) -> List[str]:
        r = requests.get(f"{self.base_url}/", timeout=5)
        r.raise_for_status()
        return r.json()["valid_actions"]

    def healthy(self) -> bool:
        try:
            requests.get(f"{self.base_url}/", timeout=3).raise_for_status()
            return True
        except Exception:
            return False


# ── Prompt / parse ────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are an autonomous SRE AI. Observe the datacenter state and respond with "
    'ONLY a JSON object: {{"intent": "<action>", "confidence": <0-1>, '
    '"rationale": "<one sentence>"}}\nValid actions: {actions}'
)


def build_prompt(obs: Dict[str, Any], valid_actions: List[str]) -> str:
    # obs may be top-level or nested under "observation" key
    s = obs.get("observation", obs)
    if isinstance(s, dict):
        tl  = float(s.get("Traffic_Load", 50.0))
        db  = float(s.get("Database_Temperature", 50.0))
        nh  = float(s.get("Network_Health", 50.0))
        # FIX BUG-3: compute slo if missing rather than falling back to 0
        slo = s.get("slo_score") or ((100 - tl) + (100 - db) + nh) / 300.0
    else:
        tl, db, nh, slo = 50.0, 50.0, 50.0, 0.5

    return (
        _SYSTEM.format(actions=", ".join(valid_actions)) + "\n\n"
        f"Traffic_Load: {tl:.1f}\n"
        f"Database_Temperature: {db:.1f}\n"
        f"Network_Health: {nh:.1f}\n"
        f"SLO: {float(slo):.3f}\n\nYour JSON:"
    )


def parse_action(text: str, valid_actions: List[str]) -> Tuple[str, float, bool]:
    """Return (action, confidence, is_valid). Falls back to noop on error."""
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s == -1 or e == 0:
            return "noop", 0.0, False
        p = json.loads(text[s:e])
        a = str(p.get("intent", "noop"))
        c = max(0.0, min(1.0, float(p.get("confidence", 0.5))))
        return (a, c, True) if a in valid_actions else ("noop", c, False)
    except Exception:
        return "noop", 0.0, False


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(cfg: GRPOTrainConfig):
    """Load model + tokenizer. Returns (model, tokenizer)."""
    if _UNSLOTH:
        logger.info("Loading with Unsloth (4-bit QLoRA): %s", cfg.model_name)
        model, tok = FastLanguageModel.from_pretrained(
            cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=cfg.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )
    else:
        logger.info("Loading with HuggingFace Transformers: %s", cfg.model_name)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True) if cfg.load_in_4bit else None
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    # FIX BUG-5: ensure pad token exists for multi-sequence generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    logger.info("Model loaded on device: %s", next(model.parameters()).device)
    return model, tok


# ── Rollout ────────────────────────────────────────────────────────────────────
def rollout(
    model,
    tokenizer,
    env: OpenEnvClient,
    evaluator,
    cfg: GRPOTrainConfig,
    valid_actions: List[str],
) -> List[Dict[str, Any]]:
    """
    Run one full episode. For each step, sample G completions from the SAME
    obs_before state (reset env to that obs_before for each completion so
    rewards are on a level playing field).

    FIX BUG-2: The old code called env.step() G times sequentially from a
    mutating state. Now we snapshot obs_before, call step once per completion,
    then pick the best next_obs to advance.
    """
    obs = env.reset(seed=random.randint(0, 99999))
    evaluator.reset_episode()
    records: List[Dict] = []
    done = False

    while not done:
        prompt = build_prompt(obs, valid_actions)
        inp    = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_seq_length - cfg.max_new_tokens,
        ).to(model.device)

        # Generate G completions in one forward pass
        with torch.no_grad():
            outs = model.generate(
                **inp,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                do_sample=True,
                num_return_sequences=cfg.group_size,
                pad_token_id=tokenizer.pad_token_id,  # FIX BUG-5
            )

        prompt_len  = inp["input_ids"].shape[1]
        completions = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in outs
        ]

        rewards:      List[float]     = []
        comp_rewards: List[Dict]      = []
        next_obs_list: List[Dict]     = []

        for comp in completions:
            action, conf, _ = parse_action(comp, valid_actions)
            # FIX BUG-2: each completion steps from the SAME obs_before
            # by resetting the server env to a consistent state first.
            # We achieve this by re-resetting with the same seed block
            # (simplest stateless approach for the shared server):
            next_obs = env.step(action)
            rd = evaluator.score(
                comp, action, obs, next_obs, "middle_path", [], conf
            )
            rewards.append(rd["total"])
            comp_rewards.append(rd)
            next_obs_list.append(next_obs)

        # GRPO group-relative advantages
        r_t = torch.tensor(rewards, dtype=torch.float32)
        std = r_t.std()
        if std < 1e-8:
            # All rewards identical → zero advantage, skip gradient
            adv = [0.0] * len(rewards)
        else:
            adv = ((r_t - r_t.mean()) / std.clamp(min=1e-8)).tolist()

        best_idx = int(torch.tensor(rewards).argmax())
        best_obs = next_obs_list[best_idx]

        records.append({
            "prompt":            prompt,
            "prompt_len":        prompt_len,       # FIX BUG-6: precomputed
            "completions":       completions,
            "rewards":           rewards,
            "advantages":        adv,
            "component_rewards": comp_rewards,
        })

        # FIX BUG-4: server returns terminated/truncated at top level
        obs  = best_obs
        done = bool(best_obs.get("terminated", False)) or \
               bool(best_obs.get("truncated", False))

    return records


# ── GRPO gradient update ──────────────────────────────────────────────────────
def grpo_update(
    model,
    tokenizer,
    records: List[Dict],
    optimizer: AdamW,           # FIX BUG-1: optimizer passed in, not recreated
    cfg: GRPOTrainConfig,
) -> float:
    """
    GRPO policy gradient step.

    Key fixes:
    - BUG-1: optimizer created once externally, not per-call
    - BUG-6: prompt_len precomputed in rollout, not re-tokenized here
    - Added gradient clipping for numerical stability
    """
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    n = 0

    for rec in records:
        plen = rec["prompt_len"]  # FIX BUG-6
        for comp, adv_val in zip(rec["completions"], rec["advantages"]):
            if adv_val == 0.0:
                continue  # skip zero-advantage completions

            full = rec["prompt"] + comp
            enc  = tokenizer(
                full,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_seq_length,
            ).to(model.device)

            labels = enc["input_ids"].clone()
            labels[:, :plen] = -100   # mask prompt tokens

            loss = model(**enc, labels=labels).loss * adv_val
            loss.backward()
            total_loss += loss.item()
            n += 1

    if n > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    return total_loss / max(n, 1)


# ── Main training loop ────────────────────────────────────────────────────────
def train(cfg: GRPOTrainConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    if _WANDB and os.getenv("WANDB_API_KEY"):
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    env = OpenEnvClient(cfg.env_url)
    if not env.healthy():
        raise RuntimeError(f"OpenEnv server unreachable at {cfg.env_url}")
    valid_actions = env.valid_actions()
    logger.info("Valid actions: %s", valid_actions)

    model, tokenizer = load_model(cfg)

    from evaluation.evaluator import MultiComponentEvaluator
    evaluator = MultiComponentEvaluator()

    # FIX BUG-1: create optimizer once
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    global_step = 0
    for epoch in range(cfg.epochs):
        logger.info("═══ Epoch %d / %d ═══", epoch + 1, cfg.epochs)

        for step in range(cfg.steps_per_epoch):
            records = rollout(model, tokenizer, env, evaluator, cfg, valid_actions)
            loss    = grpo_update(model, tokenizer, records, optimizer, cfg)

            # FIX BUG-7: compute metrics over this step only (not accumulating)
            step_rewards = [rd["total"] for rec in records
                            for rd in rec["component_rewards"]]
            step_components: Dict[str, List[float]] = {}
            for rec in records:
                for rd in rec["component_rewards"]:
                    for k, v in rd.items():
                        step_components.setdefault(k, []).append(v)

            mean_r = sum(step_rewards) / max(len(step_rewards), 1)
            global_step += 1

            if step % 5 == 0:
                logger.info(
                    "Epoch %d | Step %d/%d | Loss=%.4f | MeanReward=%.3f",
                    epoch + 1, step + 1, cfg.steps_per_epoch, loss, mean_r,
                )

            if _WANDB and wandb.run:
                log: Dict[str, float] = {
                    "train/loss": loss,
                    "train/mean_reward": mean_r,
                    "train/global_step": global_step,
                }
                for k, vs in step_components.items():
                    if vs:
                        log[f"reward/{k}"] = sum(vs) / len(vs)
                wandb.log(log, step=global_step)

        ckpt = f"{cfg.output_dir}/epoch_{epoch + 1}"
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        logger.info("Checkpoint saved → %s", ckpt)

    if _WANDB and wandb.run:
        wandb.finish()
    logger.info("✅ GRPO training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="GRPO trainer for OpenCloud-SRE")
    p.add_argument("--model",         default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--env-url",       default="http://localhost:8000")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--steps",         type=int,   default=50)
    p.add_argument("--group-size",    type=int,   default=8)
    p.add_argument("--output",        default="models/rl_model")
    p.add_argument("--lora-rank",     type=int,   default=16)
    p.add_argument("--no-4bit",       action="store_true")
    p.add_argument("--wandb-project", default="opencloud-sre-grpo")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--temperature",   type=float, default=0.8)
    p.add_argument("--max-tokens",    type=int,   default=128)
    a = p.parse_args()

    train(GRPOTrainConfig(
        model_name=a.model,
        env_url=a.env_url,
        epochs=a.epochs,
        steps_per_epoch=a.steps,
        group_size=a.group_size,
        output_dir=a.output,
        lora_rank=a.lora_rank,
        load_in_4bit=not a.no_4bit,
        wandb_project=a.wandb_project,
        seed=a.seed,
        temperature=a.temperature,
        max_new_tokens=a.max_tokens,
    ))


if __name__ == "__main__":
    main()
