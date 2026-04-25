"""
training/rl/grpo_trainer.py
============================
GRPO trainer for OpenCloud-SRE using Unsloth + trl.GRPOTrainer.

Usage:
    # Start env server first:
    uvicorn env.server:app --port 8000
    # Then run training:
    python -m training.rl.grpo_trainer --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations
import argparse, json, logging, os, random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import requests, torch

logger = logging.getLogger(__name__)

try:
    from unsloth import FastLanguageModel
    _UNSLOTH = True
except ImportError:
    _UNSLOTH = False

try:
    from trl import GRPOTrainer, GRPOConfig
    _TRL = True
except ImportError:
    _TRL = False

try:
    import wandb; _WANDB = True
except ImportError:
    _WANDB = False


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class GRPOTrainConfig:
    model_name:     str  = "Qwen/Qwen2.5-1.5B-Instruct"
    env_url:        str  = "http://localhost:8000"
    output_dir:     str  = "models/rl_model"
    epochs:         int  = 3
    steps_per_epoch:int  = 50
    group_size:     int  = 8
    max_new_tokens: int  = 256
    learning_rate:  float= 5e-6
    max_seq_length: int  = 2048
    lora_rank:      int  = 16
    load_in_4bit:   bool = True
    temperature:    float= 0.9
    wandb_project:  str  = "opencloud-sre-grpo"
    seed:           int  = 42


# ── OpenEnv client ────────────────────────────────────────────────────────────
class OpenEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/reset",
                          json={"seed": seed, "crash_on_reset": True}, timeout=10)
        r.raise_for_status(); return r.json()

    def step(self, action: str) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/step",
                          json={"action": action}, timeout=10)
        r.raise_for_status(); return r.json()

    def valid_actions(self) -> List[str]:
        return requests.get(f"{self.base_url}/", timeout=5).json()["valid_actions"]

    def healthy(self) -> bool:
        try: requests.get(f"{self.base_url}/", timeout=3).raise_for_status(); return True
        except: return False


# ── Prompt / parse ────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are an autonomous SRE AI. Observe the datacenter state and respond with "
    "ONLY a JSON object: {{\"intent\": \"<action>\", \"confidence\": <0-1>, "
    "\"rationale\": \"<one sentence>\"}}\nValid actions: {actions}"
)

def build_prompt(obs: Dict, valid_actions: List[str]) -> str:
    s = obs.get("observation", obs)
    return (
        _SYSTEM.format(actions=", ".join(valid_actions)) + "\n\n"
        f"Traffic_Load: {s.get('Traffic_Load',0):.1f}\n"
        f"Database_Temperature: {s.get('Database_Temperature',0):.1f}\n"
        f"Network_Health: {s.get('Network_Health',0):.1f}\n"
        f"SLO: {s.get('slo_score',0):.3f}\n\nYour JSON:"
    )

def parse_action(text: str, valid_actions: List[str]) -> Tuple[str, float, bool]:
    try:
        s, e = text.find("{"), text.rfind("}") + 1
        if s == -1 or e == 0: return "noop", 0.0, False
        p = json.loads(text[s:e])
        a = p.get("intent", "noop")
        c = float(p.get("confidence", 0.5))
        return (a, c, True) if a in valid_actions else ("noop", c, False)
    except: return "noop", 0.0, False


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(cfg: GRPOTrainConfig):
    if _UNSLOTH:
        model, tok = FastLanguageModel.from_pretrained(
            cfg.model_name, max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit, dtype=None)
        model = FastLanguageModel.get_peft_model(
            model, r=cfg.lora_rank,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_alpha=cfg.lora_rank*2, lora_dropout=0.05,
            bias="none", use_gradient_checkpointing="unsloth",
            random_state=cfg.seed)
        return model, tok
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True) if cfg.load_in_4bit else None
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.bfloat16)
    return model, tok


# ── Rollout ───────────────────────────────────────────────────────────────────
def rollout(model, tokenizer, env: OpenEnvClient, evaluator, cfg: GRPOTrainConfig,
            valid_actions: List[str]) -> List[Dict]:
    obs = env.reset(seed=random.randint(0, 9999))
    evaluator.reset_episode()   # clear anti-hacking action history per episode
    records = []
    done = False
    while not done:
        prompt = build_prompt(obs, valid_actions)
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outs = model.generate(
                **inp, max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature, do_sample=True,
                num_return_sequences=cfg.group_size,
                pad_token_id=tokenizer.eos_token_id)
        completions = [
            tokenizer.decode(o[inp["input_ids"].shape[1]:], skip_special_tokens=True)
            for o in outs]

        rewards, comp_rewards = [], []
        best_obs, best_r = obs, float("-inf")
        for comp in completions:
            action, conf, _ = parse_action(comp, valid_actions)
            next_obs = env.step(action)
            rd = evaluator.score(comp, action, obs, next_obs, "middle_path", [], conf)
            rewards.append(rd["total"]); comp_rewards.append(rd)
            if rd["total"] > best_r: best_r, best_obs = rd["total"], next_obs

        r_t = torch.tensor(rewards, dtype=torch.float32)
        adv = ((r_t - r_t.mean()) / (r_t.std().clamp(min=1e-8))).tolist()
        records.append({"prompt": prompt, "completions": completions,
                        "rewards": rewards, "advantages": adv,
                        "component_rewards": comp_rewards})
        obs = best_obs
        done = best_obs.get("terminated", False) or best_obs.get("truncated", False)
    return records


# ── Manual GRPO update ────────────────────────────────────────────────────────
def grpo_update(model, tokenizer, records: List[Dict], cfg: GRPOTrainConfig) -> float:
    from torch.optim import AdamW
    opt = AdamW(model.parameters(), lr=cfg.learning_rate)
    model.train(); total_loss = 0.0; n = 0
    for rec in records:
        for comp, adv in zip(rec["completions"], rec["advantages"]):
            full = rec["prompt"] + comp
            enc  = tokenizer(full, return_tensors="pt", truncation=True,
                             max_length=cfg.max_seq_length).to(model.device)
            labels = enc["input_ids"].clone()
            plen = tokenizer(rec["prompt"], return_tensors="pt")["input_ids"].shape[1]
            labels[:, :plen] = -100
            loss = model(**enc, labels=labels).loss * adv
            loss.backward(); total_loss += loss.item(); n += 1
    opt.step(); opt.zero_grad()
    return total_loss / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────────────
def train(cfg: GRPOTrainConfig) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    if _WANDB and os.getenv("WANDB_API_KEY"):
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    env = OpenEnvClient(cfg.env_url)
    if not env.healthy():
        raise RuntimeError(f"OpenEnv server unreachable at {cfg.env_url}")
    valid_actions = env.valid_actions()

    model, tokenizer = load_model(cfg)
    from evaluation.evaluator import MultiComponentEvaluator
    evaluator = MultiComponentEvaluator()

    for epoch in range(cfg.epochs):
        all_rewards: List[float] = []
        all_components: Dict[str, List[float]] = {}
        for step in range(cfg.steps_per_epoch):
            records = rollout(model, tokenizer, env, evaluator, cfg, valid_actions)
            loss = grpo_update(model, tokenizer, records, cfg)
            for rec in records:
                for rd in rec["component_rewards"]:
                    all_rewards.append(rd["total"])
                    for k, v in rd.items():
                        all_components.setdefault(k, []).append(v)
            if step % 5 == 0:
                mean_r = sum(all_rewards[-cfg.group_size:]) / cfg.group_size
                logger.info("Epoch %d | Step %d | Loss=%.4f | Reward=%.3f",
                            epoch+1, step+1, loss, mean_r)
                if _WANDB and wandb.run:
                    log = {"train/loss": loss, "train/mean_reward": mean_r}
                    for k, vs in all_components.items():
                        log[f"reward/{k}"] = sum(vs[-cfg.group_size:]) / cfg.group_size
                    wandb.log(log)

        ckpt = f"{cfg.output_dir}/epoch_{epoch+1}"
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
        logger.info("Checkpoint saved → %s", ckpt)

    if _WANDB and wandb.run: wandb.finish()
    logger.info("✅ GRPO training complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--env-url",      default="http://localhost:8000")
    p.add_argument("--epochs",       type=int, default=3)
    p.add_argument("--steps",        type=int, default=50)
    p.add_argument("--group-size",   type=int, default=8)
    p.add_argument("--output",       default="models/rl_model")
    p.add_argument("--lora-rank",    type=int, default=16)
    p.add_argument("--no-4bit",      action="store_true")
    p.add_argument("--wandb-project",default="opencloud-sre-grpo")
    a = p.parse_args()
    train(GRPOTrainConfig(
        model_name=a.model, env_url=a.env_url, epochs=a.epochs,
        steps_per_epoch=a.steps, group_size=a.group_size,
        output_dir=a.output, lora_rank=a.lora_rank,
        load_in_4bit=not a.no_4bit, wandb_project=a.wandb_project))

if __name__ == "__main__":
    main()
