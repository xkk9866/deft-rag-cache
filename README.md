# DEFT: answer cache consistency for evolving RAG corpora

This repository contains the Python implementation and experiment drivers for DEFT, a framework for deciding when cached answers to multi-hop retrieval-augmented generation (RAG) queries remain valid after corpus updates. It includes workload builders, oracle and labeling code, speculative scorers, cache policy simulation, request-level risk control, and serving benchmarks.

The repository contains **code only**. Datasets, model weights, experiment outputs, figures, and manuscript files are not included.

## Repository layout

- `common.py`, `rag.py`, `oracle.py`: shared I/O, retrieval and reader wrappers, and ground-truth traces.
- `build_hotpot.py`, `build_mquake.py`, `build_wikirev.py`: workload construction.
- `label.py`, `label_reads.py`, `margins.py`, `mcert.py`: training labels and reader-margin measurements.
- `deft.py`, `deltar.py`, `spec*.py`: speculative models and certification components.
- `simulate.py`, `request_risk.py`, `vcache_alg.py`, `baselines.py`: replay policies, risk control, and baselines.
- `run_eval.py`, `run_eval3.py`, `bench*.py`, `serve_bench*.py`: evaluation and serving benchmarks.
- `run_*.sh`: historical experiment recipes. Many contain paths and GPU IDs from the original Linux server; edit them for your environment before use.

## Environment

Use Linux with a CUDA GPU for the full pipeline. The Python code uses NumPy, PyTorch, Transformers, vLLM, FAISS, pandas, PyArrow, SciPy, scikit-learn, and `huggingface_hub`. Install versions compatible with your CUDA driver and each other. No exact environment lockfile is available, so results may vary across versions and hardware.

Set the shared paths before running a stage:

```bash
export EVROOT=/path/to/deft-workspace
export BGE=/path/to/bge-base-en-v1.5
export LLM_PATH=/path/to/reader-model
mkdir -p "$EVROOT/data" "$EVROOT/runs" "$EVROOT/logs"
```

`common.py` reads these environment variables. Some workload and legacy run scripts also contain absolute paths; inspect the chosen script and replace those paths before running it. External datasets and model weights must be obtained separately under their own terms.

## Typical workflow

Run the Python programs from the repository root so local imports resolve:

```bash
python build_hotpot.py --help
python oracle.py --help
python deft.py --help
python run_eval3.py --help
```

The general order is: prepare a workload, produce oracle traces for train/calibration/test splits, derive labels and margins, train or load a speculative model, then run a replay or serving benchmark. The exact commands and available workload names are defined in the corresponding scripts; `run_*.sh` files record concrete experiment configurations.

The small request-risk regression check can be run without the GPU pipeline:

```bash
python -m unittest test_request_risk.py
```

## Scope and reproducibility

This is research code, including exploratory and historical experiment variants. The full results require the original workload data, model checkpoints, GPU environment, and script-specific settings. Generated outputs should remain outside this repository.
