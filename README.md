# 🚀 OpenCloud-SRE
**A Cognition-Efficient Autonomous Incident Response Architecture**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Simulation_Engine-EE4C2C?logo=pytorch)
![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-brightgreen)
![FAISS](https://img.shields.io/badge/Vector_Memory-FAISS-purple)
![Unsloth](https://img.shields.io/badge/RL-Unsloth_%7C_GRPO-orange)
![Meta](https://img.shields.io/badge/Hackathon-Meta_PyTorch_x_HuggingFace-blue)

**Team:** VICODE O(1)  
**Target Themes:** Multi-Actor Environments (Theme 1) & Enterprise Workflows (Theme 3.1)

---

## 🛑 The Core Problem
Enterprise incident response fails under pressure because AI reasoning is expensive, slow, and noisy. Standard multi-agent frameworks scale linearly in cost: agents write paragraphs to debate solutions, hallucinate dependencies, and burn heavy compute just to resolve known issues.

**OpenCloud-SRE** scales by compressing cognition. We built a system that minimizes unnecessary LLM compute while mathematically guaranteeing infrastructure safety through deterministic risk filters and O(1) memory retrieval.

## ⚙️ The Innovation: Cognitive Compression Architecture
We built a high-speed, tensor-based PyTorch simulation of an enterprise data center, wrapped in the **OpenEnv** FastAPI standard. The environment enforces **strict partial observability** (Network Node sees traffic; DB Node sees heat).

To resolve outages, our LangGraph orchestrator passes every state through 4 layers of Cognitive Compression:

1. **Incident DNA Memory (The Fast Path):** Hashes the PyTorch crash state into a vector. FAISS matches it against past resolutions for instant, O(1) execution. *Zero LLM compute used.*
2. **Shadow Consensus (The Middle Path):** Instead of deep ChatOps, agents exchange 20-byte JSON "micro-intents" (e.g., `{"intent": "throttle"}`). The Lead SRE routes for synergy, reducing coordination latency by 90%.
3. **Predictive Blast Radius (Deterministic Safety):** Proposed actions are checked against a hardcoded Dependency Matrix. Actions that trigger cascading failures are blocked before execution, neutralizing LLM hallucinations.
4. **Adaptive Trust Layer (Escrow Execution):** If AI confidence drops below 90%, the system halts, stages the Python fix in "Execution Escrow," and pages a human via the UI for a single-click approval.

## 🧠 The MLOps & Training Stack (Hackathon Alignment)
We aren't just hooking up APIs; we are building an RL-compatible environment to train domain-specific open-source models.

* **The Environment:** Standardized via `OpenEnv` client-server architecture and deployed as a Hugging Face Docker Space.
* **Adversarial Fault Injection:** GPT-4o acts as a Chaos Monkey, dynamically corrupting the PyTorch tensor.
* **Anti-Hacking Reward Function:** We use discrete, deterministic reward columns (`blast_radius_penalty`, `state_recovery_reward`, `format_reward`) instead of a single LLM judge to prevent reward hacking.
* **The Optimizer (GRPO):** We utilize **Unsloth** for highly efficient 4-bit LoRA loading and **Hugging Face TRL's GRPOTrainer** (Group Relative Policy Optimization) to continuously drive down Mean Time To Recovery (MTTR) over time.

## 📊 The "War Room" Dashboard
Our Streamlit dashboard proves the architecture in real-time. It tracks:
* Live PyTorch Server Telemetry (Traffic vs. Heat)
* **Tokens / Compute Saved** via the DNA Memory and Shadow Consensus.
* Escrow Execution terminal waiting for Human-in-the-Loop approval.

---

## 🛠️ Quick Start (Local Reproduction)

To reproduce our environment and kick off the MLOps pipeline on a machine with GPU access:

### 1. Setup & Dependencies
```bash
git clone [https://github.com/YOUR_USERNAME/OpenCloud-SRE.git](https://github.com/YOUR_USERNAME/OpenCloud-SRE.git)
cd OpenCloud-SRE
python3 -m venv venv
source venv/bin/activate

# Install Core & Unsloth
pip install torch torchvision xformers --default-timeout=1000
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install -r requirements.txt
