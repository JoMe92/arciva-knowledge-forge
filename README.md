# Arciva Knowledge Forge – LLM Assistant Proof of Concept

## Project Overview
Arciva is a cross-platform photo library manager built with a FastAPI backend, React frontend, and a Rust-based image-processing core. This repository contains the proof-of-concept notebook used to validate an in-app assistant that can answer product-specific questions, guide photographers through Arciva’s workflows, and surface contextual help. The prototype was first developed on Google Cloud/Colab and later adapted for RunPod so it can run in both managed and self-hosted GPU environments.

## What’s Included
- `arciva_assistant_poc.ipynb` – the primary notebook documenting the end-to-end workflow (environment bootstrap, data processing, QLoRA fine-tuning, evaluation, dashboarding, and artifact export).
- `arciva_assistant_poc.py` – script-exported version of the notebook for reference or automation.
- `train_arciva_assistant_local.py` – CLI entry point for running the training loop outside notebooks.
- `train_arciva_assistant_vertex.py` – Vertex AI-friendly variant that pulls data from GCS and runs on managed GPUs.
- `data/` – placeholder for local datasets (the default run expects `arciva_qa_synthetic.jsonl`).
- `requirements.txt` – package list when you need to recreate the environment outside of Colab/RunPod.
- `requirements.vertex-ai.txt` – dependency lock for building the Vertex AI training container.

## Key Capabilities Demonstrated
1. **Rapid Environment Recreation** – single cell installs pin all Python dependencies; restart-and-rerun is enough to recover from import issues.
2. **Dataset Handling** – synthetic Arciva Q&A corpus (≈1.2k items) is loaded, schema-checked, and split into train/validation/test partitions.
3. **QLoRA Fine-Tuning** – adapters are applied to a 7B base model for efficient instruction tuning; a TinyLlama fallback supports quick smoke tests.
4. **Monitoring & Evaluation** – live loss plotting, perplexity/BLEU/ROUGE metrics, and qualitative dashboards give early feedback on assistant quality.
5. **Artifact Preparation** – trained adapters are merged into fp16 base weights, ensuring the resulting checkpoint is portable to FastAPI or edge deployments.

## Running the Notebook
1. **Open environment**
   - *Google Colab*: Upload `arciva_assistant_poc.ipynb` or open directly from GitHub.
   - *RunPod/Jupyter*: Clone the repo and launch JupyterLab, ensuring GPU access is enabled.
2. **Execute Cell 1 (“Environment Configuration”)**
   - Installs all required packages (`typing_extensions`, `pydantic`, Transformers stack, etc.). Keep all pip commands in that cell for reproducibility.
3. **Run Cell 2 (“Library Imports & Global Configuration”)**
   - Confirms hardware availability and exposes toggles such as `USE_DEBUG_MODEL`, global hyperparameters (epochs, learning rate, sequence length, batch sizes, logging cadence), and paths.
4. **Proceed through the remaining sections in order**
   - Data ingestion, training, evaluation, dashboard generation, and artifact merge each rely on outputs from the previous steps.

> **Tip:** For fast debugging on small GPUs, set `USE_DEBUG_MODEL = True` to swap the base model to `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (ships with `safetensors`, so it avoids older `torch.load` vulnerabilities). Switch back to `False` before full-scale runs.

## Dataset Strategy
- **Synthetic seed**: `arciva_qa_synthetic.jsonl` mimics typical Arciva queries (“How do I tag…?”, “Where can I find…?”). It exists to verify the training pipeline, not for production-quality results.
- **Planned improvements**:
  - Clean and correct current samples via the internal labeling tool.
  - Expand to 6k+ high-quality question/answer pairs with realistic phrasing and edge cases.
  - Reduce hallucination-prone answers and improve coverage of tagging/search/export workflows.

## Evaluation & Dashboarding
- Quantitative metrics: Perplexity, BLEU, and ROUGE-L are computed on the held-out test set to provide sanity checks before investing in larger datasets.
- Interactive review: A simple dashboard (DataTable if running in Colab, pandas preview elsewhere) highlights best/worst generations per split to guide iterative data fixes.

## Deployment Direction
- After fine-tuning, adapters are merged into fp16 base weights (`model_output/` → `final_merged_model/`), which serves as the starting point for:
  - Further quantization (4-bit/Int8) for Jetson-class devices.
  - Latency profiling and FastAPI integration.
  - Edge or on-device hosting scenarios aligned with Arciva’s performance targets.

## Roadmap
1. Replace synthetic data with labeled conversations from the Arciva tagging tool.
2. Integrate automated evaluation suites (e.g., task-specific checklists, hallucination probes).
3. Package the inference stack into a deployable FastAPI microservice backed by the merged checkpoint.
4. Harden the RunPod workflow (startup scripts, monitoring) so it mirrors the eventual production environment.

Feel free to fork the repository, adapt the notebook to your own assistant ideas, and report issues or suggestions that make the Arciva LLM workflow clearer.
