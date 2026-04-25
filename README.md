# 🚀 OpenCloud-SRE
**The Autonomous Incident Command for Data Cloud Systems**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Simulation_Engine-EE4C2C?logo=pytorch)
![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-brightgreen)
![FAISS](https://img.shields.io/badge/Vector_Memory-FAISS-purple)
![Unsloth](https://img.shields.io/badge/RL-Unsloth_%7C_GRPO-orange)
![Meta](https://img.shields.io/badge/Hackathon-Meta_PyTorch_x_HuggingFace-blue)

**Team:** VICODE O(1)  
**Target Themes:** Multi-Actor Environments (Theme 1) & Enterprise Workflows (Theme 3.1)

---

## 🛑 The Problem: The AI Scaling Crisis
Modern SRE teams are overwhelmed, but traditional multi-agent AI frameworks fail in high-stakes infrastructure for three reasons:
1.  **Too Expensive:** Standard LLMs scale linearly in cost with every alert.
2.  **Too Slow:** Deep-reasoning agents take minutes to "debate" a fix while servers burn.
3.  **Too Dangerous:** LLM "hallucinations" can turn a minor database glitch into a total system blackout.

## 🏛️ The Innovation: Cognitive Compression Architecture
**OpenCloud-SRE** replaces expensive, token-heavy LLM reasoning with a multi-tiered routing architecture. We scale cloud reliability by compressing cognition—minimizing unnecessary LLM compute while mathematically guaranteeing infrastructure safety.

Every incident passes through our optimized pipeline:

1. **Incident DNA Memory (The Fast Path):** Hashes the crashed system state into a 3D vector. FAISS matches it against past resolutions. If a safe match exists, we execute instantly. *Zero LLM compute used. O(1) search time.*
2. **Shadow Consensus Swarm (The Middle Path):** Instead of chatty paragraphs, specialized Network and Database agents exchange 20-byte JSON "micro-intents" under partial observability. A Lead SRE node uses a Synergy Matrix to find the winning action, reducing latency by 90%.
3. **ChatOps Deep Negotiator (The Slow Path):** Only triggered during severe logic conflicts. This is the *only* layer where heavy tokens are spent on deep reasoning.

## 🛡️ Governance: The "Hand of God" Safety Layers
AI shouldn't have unchecked root access. We built two deterministic safety nets:

* **Predictive Blast Radius Filter:** Every proposed action is checked against a hardcoded Dependency Matrix. The system knows `circuit_breaker` is normally safe, but *CRITICAL* if the DB is failing over. Cascading failures are blocked before execution.
* **Adaptive Trust Layer (ATL):** If the AI's confidence score drops below `0.90`, the system halts, stages the Python fix in "Execution Escrow," and pages a human via the UI for a single-click approval.

---

## 🧠 The Environment & MLOps Stack
We aren't just hooking up APIs; we built a 100% open-source, RL-compatible environment to train domain-specific models.

* **The Simulation:** A high-speed, tensor-based PyTorch simulation of an enterprise data center, wrapped in the **OpenEnv** FastAPI standard. 
* **The Chaos Engine:** A high-speed stochastic Python script dynamically injects complex, multi-variable faults into the PyTorch tensors.
* **The Optimizer (GRPO):** We utilize **Unsloth** for highly efficient 4-bit LoRA loading and **Hugging Face TRL's GRPOTrainer** to continuously drive down Mean Time To Recovery (MTTR).

### 🛑 Defending Against RL Reward Hacking
We use a discrete, deterministic reward function to prevent the model from gaming the system. Our checks include:
* **Noop Abuse Penalty (-30):** Penalizes the agent for doing nothing while the system burns just to avoid blast-radius penalties.
* **Plausibility Validator (-40):** Catches physically impossible state transitions to prevent environment exploitation.
* **Confidence Calibration (-15):** Forces honest self-assessment to ensure the ATL triggers correctly.

---

## 🛠️ Quick Start (Local Reproduction)

To reproduce our environment and kick off the MLOps pipeline on an Ubuntu/Linux machine with GPU access:

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
```

### 2. Run the OpenEnv Server (CPU)
In Terminal 1, boot the PyTorch environment behind the FastAPI wrapper:
```bash
uvicorn env.server:app --host 0.0.0.0 --port 7860
```

### 3. Launch the "War Room" UI (CPU)
In Terminal 2, start the Streamlit incident command dashboard to visualize the live telemetry and Execution Escrow:
```bash
streamlit run ui/app.py
```

### 4. Execute the Training Pipeline (GPU)
To watch the model learn via our verifiable GRPO loop:
```bash
# Generate the synthetic SFT warm-up data (Uses Llama-3 via HF)
python3 -m training.sft.dataset_generator --count 50

# Run Supervised Fine Tuning (Unsloth + TRL)
python3 training/sft/train_sft.py

# Execute the Group Relative Policy Optimization (GRPO) Loop
python3 training/rl/grpo_trainer.py
```
```

***
