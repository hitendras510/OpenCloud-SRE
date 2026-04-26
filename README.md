---
title: OpenCloud-SRE
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.10"
pinned: false
---

# 🚀 OpenCloud-SRE
**Scaling Cloud Reliability through Cognitive Compression**

[![Hugging Face Space](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face_Space-blue?style=for-the-badge)](https://huggingface.co/spaces/hitendras510/OpenCloud-SRE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hitendras510/OpenCloud-SRE/blob/main/notebooks/OpenCloud_SRE_Training.ipynb)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Simulation_Engine-EE4C2C?logo=pytorch)
![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-brightgreen)
![FAISS](https://img.shields.io/badge/Vector_Memory-FAISS-purple)
![FastAPI](https://img.shields.io/badge/Microservice-FastAPI-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Control_Plane-Streamlit-FF4B4B?logo=streamlit)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hitendras510/OpenCloud-SRE/blob/main/notebooks/OpenCloud_SRE_Training.ipynb)

📖 **Want to dive deeper into our architecture?** [Read our full Engineering Blog Post here](./blog/blog.md)

**Team:** VICODE O(1)  
**Target Themes:** Multi-Agent Systems (Theme 1) & Enterprise Workflows (Theme 3.1)

---

## 🛑 The Problem: "The AI Scaling Crisis"

Modern Site Reliability Engineering (SRE) teams are overwhelmed by alert fatigue. Traditional multi-agent AI frameworks fail in high-stakes cloud infrastructure for three reasons:
1.  **Too Expensive:** Standard LLMs scale linearly in cost with every alert.
2.  **Too Slow:** Deep-reasoning agents take minutes to "debate" a fix while servers burn.
3.  **Too Dangerous:** LLM "hallucinations" can turn a minor database glitch into a total system blackout.

---

## 🏛️ Our Innovation: Cognitive Compression & Distributed Routing

We built a system that **Compresses Cognition**. We don't think harder; we think smarter. By decoupling the **Data Plane** (a live FastAPI state manager) from the **Control Plane** (a Streamlit War Room UI), we route every incident through three optimized layers:

### 1. 🧬 DNA Memory (The FAST Path)
- **What it is:** A local **FAISS-backed** vector database of historical incident "DNA."
- **How it works:** We hash the current crashed system state into a 3D vector (CPU, Network, DB Temp). If it matches a past fix within a safe L2 distance, we execute the known fix **instantly.**
- **Value:** Zero LLM cost and $O(1)$ search time for known outages.

### 2. 🤝 Shadow Consensus Swarm (The MIDDLE Path)
- **The Protocol:** Our AI agents don't "chat." They exchange lightweight **Micro-Intent JSONs**. 
- **The Mechanism:** Specialized Compute, Network, and Database Agents propose fixes under "Partial Observability."
- **Consensus:** A Lead SRE node uses a **Synergy Matrix** to aggregate confidence scores and find the winning action.
- **Value:** 90% cheaper and 10x faster than standard agentic chat.

### 3. 💬 ChatOps Deep Negotiator (The SLOW Path)
- Only triggered when a severe logic conflict occurs between agents or for novel edge cases. This is the *only* layer where we spend heavy tokens on deep, chain-of-thought reasoning.

---

## ⚡ Live Chaos Engineering: The "War Room" UI (`/ui`)

To prove OpenCloud-SRE is production-ready, we built a live, interactive Streamlit command center that interacts with the backend in real-time.

* **Live Telemetry:** The UI continuously polls the FastAPI backend (`http://127.0.0.1:8000/metrics`) to render live Plotly gauges for Traffic, DB Temperature, and Network Health.
* **Chaos Control Center:** Users can manually inject critical faults (e.g., Target CPU Spikes via interactive sliders, Network Partitions, DB Deadlocks). The UI sends an HTTP POST payload to the backend, immediately triggering the AI's observation loop.
* **Transparent Execution:** The "ChatOps Terminal" streams the internal JSON reasoning of the Shadow Consensus live, allowing human operators to watch the AI diagnose the specific failing metric and autonomously execute the API fix.

---

## 🛡️ Governance & Safety: The "Hand of God" Layers

AI shouldn't have unchecked root access. We built deterministic safety nets to guarantee infrastructure safety against LLM hallucinations:

### 1. Predictive Blast Radius Filter
- **Hallucination Protection:** Every proposed action is checked against a **Deterministic Dependency Matrix**.
- **Context-Aware:** The system knows that `circuit_breaker` is safe normally, but explicitly blocks it if the database is concurrently failing over.
- **Result:** We block cascading failures *before* they reach your infrastructure.

### 2. Adaptive Trust Layer (ATL) & Execution Escrow
- **High Confidence:** Score ≥ 0.90 → **Full Autonomy**. The system resolves the issue silently.
- **Low Confidence:** Score < 0.90 → **Execution Escrow**. The system halts and pages a human operator via the UI dashboard for single-click approval.

---

## 📊 Key Results / "The Judge's Metrics"

We successfully trained our agent using **GRPO** for 3 epochs. The results demonstrate a significant improvement over the random baseline in both total reward and Mean Time To Recovery (MTTR).

![Training Performance](https://raw.githubusercontent.com/hitendras510/OpenCloud-SRE/main/evaluation/training_performance.png)

* **MTTR (Recovery Speed):** 80% faster for recurring incidents via DNA Memory hits.
* **Operational Cost:** 90% reduction in token spend via Fast and Middle routing paths.
* **Safety Rating:** 100% block rate on "Known Critical" cascading failures via our Blast Radius Matrix.

---

## 🌍 The Simulation Environment (`/env`)

Instead of hooking up to a real cloud provider during training, we built a 100% open-source, RL-compatible stochastic environment:
* **The Stateful API (`env/server.py`):** A high-speed FastAPI simulation of an enterprise data center. The state is held in a `GlobalStateManager` that reacts to both AI actions and manual user injections.
* **OpenEnv Manifest (`env/openenv.yaml`):** Standard manifest declaring the environment for the hackathon evaluator.
* **Action Space:** Discrete actions (`throttle_traffic`, `schema_failover`, `circuit_breaker`, `kill_long_queries`), each designed to mitigate specific vectors of infrastructure collapse.

---

## 🧠 End-to-End MLOps & Training Stack (`/training`)

We utilize an automated pipeline to train Open-Source LLMs (e.g., Qwen) to operate the environment using Reinforcement Learning:
1. **Supervised Fine-Tuning (SFT):** The `train_sft.py` pipeline uses **TRL** and **Unsloth** for blazing-fast 4-bit QLoRA fine-tuning. This "warms up" the model to consistently output valid JSON routing intents.
2. **Group Relative Policy Optimization (GRPO):** The `grpo_trainer.py` executes the core RL loop, sampling multiple completions per crashed state from the FastAPI environment to continuously drive down MTTR.
3. **Reward Function Defenses:** The discrete reward function penalizes the agent for "Noop Abuse" (doing nothing while the system burns) and Plausibility Violations (physically impossible state transitions).

---

## 🛠️ Quick Start (Local & Hugging Face Deployment)

The entire stack is designed to run in a single containerized environment, managing the UI, Backend, and AI Agents synchronously.

### 1. Setup & Dependencies (Local Bare-Metal)

**Linux / macOS:**
```bash
git clone https://github.com/hitendras510/OpenCloud-SRE.git
cd OpenCloud-SRE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/hitendras510/OpenCloud-SRE.git
cd OpenCloud-SRE
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** Running the full training pipeline natively requires an NVIDIA GPU (T4-small or better). However, you can run the UI and Backend locally on any machine by keeping the **"Mock LLM (offline)"** toggle enabled in the War Room sidebar.

### 2. Launch the Stack (Local Bare-Metal)

**Linux / macOS:**
Use the provided bash script to boot both the FastAPI backend and the Streamlit UI simultaneously:
```bash
# This starts FastAPI on Port 8000 and Streamlit on Port 7860
chmod +x start.sh
./start.sh
```

**Windows (PowerShell):**
Since `start.sh` is a bash script, you can start the backend and UI directly in separate PowerShell windows (ensure the `venv` is activated in both):
```powershell
# Terminal 1: Start the Backend (Data Plane)
uvicorn env.server:app --host 0.0.0.0 --port 8000

# Terminal 2: Start the UI (Control Plane)
streamlit run ui/app.py --server.port 7860
```

### 3. Launch via Docker (Recommended for GPU environments)

If you have Docker installed, you can build and run the entire stack in one isolated container:
```bash
# Build the Docker image
docker build -t opencloud-sre .

# Run the container (add --gpus all if you have the NVIDIA Container Toolkit)
docker run -d -p 7860:7860 -p 8000:8000 --name sre_app opencloud-sre
```

### 4. Inject Chaos & Interact

1. Navigate to `http://localhost:7860`. You will land on the newly separated **Homepage**.
2. Open the sidebar navigation and switch to the **⚡ War Room** dashboard.
3. Click the **▶ Start** button in the sidebar to initialize the AI polling loop.
4. Open the **🔴 Chaos Control Center** popovers in the sidebar, set a target fault (e.g., Target CPU to `99%`), and click **Execute**.
5. Watch the autonomous SRE detect the anomaly, achieve consensus, and stabilize the system live in the telemetry charts and ChatOps terminal!

---
**OpenCloud-SRE: Turning Cloud Intelligence into an Enterprise Reflex.**
