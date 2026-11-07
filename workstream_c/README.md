# Workstream C: Live End-to-End Evaluation

This directory contains the live evaluation harness, experimental runners, and analysis pipelines for evaluating the Handoff PRM against downstream multi-agent execution.

## Requirements and Offline Setup
- **Hugging Face Model Cache**: The SentenceTransformer model (`all-MiniLM-L6-v2`) used in feature extraction must be cached locally in your Hugging Face cache directory (`~/.cache/huggingface/hub`) because `feature_pipeline.py` and the live runners execute in offline mode (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- **Dependencies**: Python 3.10.7 with packages listed in repo root `requirements.txt`.
- **API Keys**: Configured in root `.env` as `GEMINI_API_KEY_1` through `GEMINI_API_KEY_18`.

