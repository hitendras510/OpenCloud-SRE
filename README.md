# 🚀 OpenCloud-SRE
**Autonomous Multi-Node Incident Command for Data Cloud Systems**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Simulation_Engine-EE4C2C?logo=pytorch)
![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-brightgreen)
![Unsloth](https://img.shields.io/badge/RLHF-Unsloth_%7C_TRL-orange)
![Meta](https://img.shields.io/badge/Hackathon-Meta_PyTorch_x_HuggingFace-blue)

**Team:** VICODE O(1)  
**Target Themes:** Multi-Agent Interactions (Halluminate Bonus) & Professional Tasks (Scaler AI Labs Bonus)

---

## 🛑 The Problem
When a massive enterprise data pipeline crashes, a single AI agent cannot fix it because it lacks complete visibility. Real incident resolution requires a "War Room"—a team of specialized Site Reliability Engineers communicating under pressure. 

Current LLM environments evaluate "toy" problems like grid-worlds. **OpenCloud-SRE** tackles high-stakes, real-world Data Cloud Engineering. We built a system to train open-source models to execute enterprise infrastructure recovery.

## ⚙️ The Innovation: Tensor-Based OpenEnv Simulation
Instead of building a slow, clunky web simulator, we built the entire data cloud environment natively in **PyTorch** within the **OpenEnv** framework. 
* **The Math:** The health of the servers, API gateways, and database connection pools are represented mathematically as state tensors (`Traffic_Load`, `Database_Temperature`, `Network_Health`). This allows the environment to run thousands of simulation steps per second during RL training.
* **Partial Observability:** The environment enforces strict blind spots. The Network Controller can only see traffic spikes; the Database Controller can only see slow query logs. Neither can resolve the outage alone.

## 🧠 The Architecture

1. **The SRE Cluster (LangGraph):** A multi-node communication bus managed by a Lead SRE (Orchestrator). The Network and Database controllers must negotiate a recovery strategy and execute strictly typed Python commands (e.g., `execute_throttle()`, `schema_failover()`) in the correct sequence to avoid cascading failures.
2. **The Chaos Monkey (GPT-4o):** We utilize the OpenAI API as an adversarial chaos engine, dynamically injecting complex, multi-variable faults into the PyTorch environment tensor to create infinite training scenarios.
3. **The Automated Judge (GPT-4o):** Acts as a strict "Senior Engineer," scoring the cluster's Python diagnostic logs from -100 to +100 for safety and efficiency.
4. **The Training Pipeline (Unsloth + TRL):** We use Supervised Fine-Tuning (SFT) to warm up the models, followed by Proximal Policy Optimization (PPO) via Hugging Face TRL to align the models to the environment's reward signals.

## 📊 The "War Room" Dashboard
To make the invisible math visible, OpenCloud-SRE includes a lightning-fast **Streamlit** dashboard. It provides live, split-screen telemetry showing the PyTorch server health crashing and recovering alongside the real-time LangGraph message bus.

---

## 🛠️ Quick Start (For Hackathon Compute / VM)

To reproduce our environment and kick off the training loop on an Ubuntu/Linux machine with GPU access:

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/OpenCloud-SRE.git](https://github.com/YOUR_USERNAME/OpenCloud-SRE.git)
cd OpenCloud-SRE

# 2. Set up the virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 3. Install PyTorch first (to prevent dependency conflicts)
pip install --default-timeout=1000 torch torchvision xformers

# 4. Install Unsloth directly from source
pip install --default-timeout=1000 "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"

# 5. Install the OpenEnv, routing, and UI plumbing
pip install trl langgraph streamlit openai python-dotenv openenv wandb6666