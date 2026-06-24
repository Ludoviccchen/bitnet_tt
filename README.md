# BitNet 1.58B Inference on Tenstorrent Blackhole P150a

## Project Description

This repository contains the research and engineering work aimed at implementing **BitNet 1.58B** algorithms for highly efficient LLM inference. The project focuses on integrating a 1.58-bit weight representation (ternary values: -1, 0, 1) into lightweight inference environments such as `llm.c` and `llama.cpp`.

Hardware optimization is at the core of this project: BitNet-style matrix operations are optimized at the kernel level using **Tenstorrent APIs (TT-Metal)** to fully leverage the architecture of the **Tenstorrent Blackhole P150a** hardware.

### Main Objectives
*   **Weight Representation**: Implement 1.58-bit weights and manage quantized linear layers.
*   **Kernel Development**: Create custom kernels (C++/RISC-V) via TT-Metalium for ternary matrix multiplication operations.
*   **Integration**: Deploy the optimized kernels into custom forks of `llm.c` and `llama.cpp`.
*   **Benchmarking**: Strictly evaluate throughput, latency, memory usage, and accuracy against a standard FP16/INT8 baseline.

---

## Repository Structure

The project is organized to clearly separate experiments, low-level code, and evaluation tools:

*   `src/`: Python scripts (Tenstorrent API validation tests, quantization logic) and C++ wrappers.
*   `kernels/`: Source code for low-level kernels specific to the Blackhole p150a RISC-V architecture.
*   `benchmarks/`: Performance evaluation scripts (latency, throughput, memory profiling) and raw results.
*   `docs/`: Internal documentation, progress reports, and final presentation.
*   `models/`: *(Git-ignored)* Local directory to store tested GGUF or binary models.
*   `third_party/`: Git submodules containing modified forks of `llm.c` and `llama.cpp`.

---

## System Requirements

To run this project, the following environment is required:
*   **Hardware:** Server equipped with a Tenstorrent Blackhole p150a PCIe card.
*   **Drivers:** Tenstorrent drivers installed and recognized via the `tt-smi` command.
*   **Environment:** Python 3.12 (via Anaconda/Miniconda) and a modern C++ compiler (Clang/GCC).
*   **Low-level Dependencies:** Source installation of [TT-Metalium](https://github.com/tenstorrent/tt-metal).

---

## Installation & Getting Started

### 1. Cloning the Repository
Since this project relies on submodules, it is mandatory to use the `--recursive` flag when cloning:
```bash
git clone --recursive https://github.com/ludoviccchen/bitnet_tt
cd bitnet-tt-blackhole
```

### 2. Conda Environment Setup
```bash
conda create -n bitnet_tt python=3.12 -y
conda activate bitnet_tt
```

### 3. Linking with TT-Metal
```bash
export TT_METAL_HOME="/path/to/your/tt-metal/installation"
export PYTHONPATH="${TT_METAL_HOME}"
```