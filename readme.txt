# Adaptive_Beamforming

**Adaptive Beamforming for Interference-Limited MU-MIMO using Spatio-Temporal Policy Networks**

This project introduces a machine learning–based framework for receive-side beamforming in MU-MIMO wireless systems. It is designed for environments with partial Channel State Information (CSI), user mobility, and interference-limited conditions—common in consumer-grade Wi-Fi and 5G NR deployments.

The proposed method consists of:
- A **CNN-GRU encoder** that learns spatio-temporal CSI features.
- A **Reinforcement Learning (PPO) agent** that dynamically estimates a quantized combining matrix to maximize user SINR or throughput.

The system outperforms traditional methods like MRC, MMSE, and RBD by adaptively learning from past channel conditions and generalizing across time-varying environments.

---

## 🚀 Features
- Two-stage learning: Spatio-temporal encoder + RL-based combiner
- Quantization-aware combining matrix
- PPO agent with SINR-driven reward
- Full MU-MIMO simulation in TensorFlow using the Sionna library
- Evaluation against classical baselines (MMSE, RBD, etc.)

---

## 🐳 Docker Setup

### 1. Build Docker Image (No Cache)
```bash
docker build --no-cache -t adaptive_beamforming .

### 2. Build Docker Image (No Cache)
```bash
docker run --rm -it --gpus all -v "$(pwd):/workspace" adaptive_beamforming /bin/bash

### 3. Build Docker Image (No Cache)
```bash
python3 main.py
