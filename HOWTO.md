# RAG Optimization Project - Lab 3 & Lab 4 Guide

This document explains how to set up and run the experiments for Lab 3 and Lab 4. The project structure has been organized for clarity and ease of use.

## Directory Structure
- `lab3/`: Contains code and scripts for Lab 3.
- `lab4/`: Contains code and scripts for Lab 4 experiments.
- `data/`: Expected location for dataset files (e.g., `dev.jsonl`). **Note:** Ensure your data is placed here.
- `bm25_wiki_index/`: Expected location for the Wikipedia BM25 index. **Note:** Ensure your index is placed here or in the lab folders if generated there.

## Prerequisites

### 1. Environment Variables
The scripts require your Hugging Face token to access models like Llama-3.1.
You **MUST** export this variable before running any python script or submitting a slurm job.

**Option A: Export in terminal (for interactive sessions)**
```bash
export HF_TOKEN="your_hugging_face_token_here"
```

**Option B: Edit Slurm Scripts**
The provided `part3.slurm` files in both `lab3/` and `lab4/` have a placeholder for this token.
Open `lab3/part3.slurm` or `lab4/part3.slurm` and update the line:
```bash
export HF_TOKEN="Your_HF_Token_Here"
```

### 2. Data Files
The scripts look for data relative to the script location.
- **Dev Data**: Expected at `../data/dev.jsonl` (relative to the script) or `data/dev.jsonl` (if running from root). The code attempts to find it at `data/dev.jsonl` relative to the script's directory, so you might need to adjust symlinks or move files if your data is elsewhere.
    - *Default configured path:* `os.path.join(os.path.dirname(__file__), "data/dev.jsonl")`
    - *Action:* Ensure `lab3/data/dev.jsonl` and `lab4/data/dev.jsonl` exist, OR symlink them from a central location.

- **Wiki Index**: Expected at `bm25_wiki_index/`.
    - *Default configured path:* `os.path.join(os.path.dirname(__file__), "bm25_wiki_index/")`
    - *Action:* Ensure the index exists in `lab3/bm25_wiki_index/` and `lab4/bm25_wiki_index/`.

## Running Lab 3

1. Navigate to the `lab3` directory:
   ```bash
   cd lab3
   ```
2. Ensure your `HF_TOKEN` is set.
3. Submit the slurm job:
   ```bash
   sbatch part3.slurm
   ```
   *Note: This runs `part3_a.py` by default.*

## Running Lab 4

1. Navigate to the `lab4` directory:
   ```bash
   cd lab4
   ```
2. Edit `part3.slurm` to uncomment the specific experiment script you want to run (default is `part3_a.py`).
   Available scripts:
   - `part3_a.py`
   - `experiment_2.py`
   - `ms_marco.py`
   - `ms_marco_l6.py`
   - `qnli_electra_base.py`
   - `qnli_sys_prompt.py`
3. Submit the slurm job:
   ```bash
   sbatch part3.slurm
   ```

## Troubleshooting
- **Token Errors**: Double-check that `HF_TOKEN` is exported or set in the slurm script.
- **File Not Found**: Check that `data/` and `bm25_wiki_index/` folders are present inside `lab3/` and `lab4/`, or update the paths in the python scripts if you prefer a shared location.
