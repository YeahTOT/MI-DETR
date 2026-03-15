# Repository Guidelines

## Project Structure & Module Organization
Core training and inference code lives in `ultralytics/`, with public entry scripts at the repo root such as `train.py`, `val.py`, `pred.py`, `export.py`, and the ONNX video helpers. Model configs are under `improve_multimodal/our_resnet18_brain/`. Keep downloadable artifacts in `checkpoints/`, dataset layouts in `datasets/`, and documentation in `docs/`. Treat `runs/` as generated output for training, validation, and inference, not source code.

## Build, Test, and Development Commands
Use the checked-in virtual environment: `source .venv/bin/activate`. Install dependencies with `pip install -r requirements.txt`. Common workflows:

```bash
python train.py --data data.yaml --dataset-root /path/to/DAUB-R_retina --device 0
python val.py --weights checkpoints/DAUB-R.pt --data data.yaml --dataset-root /path/to/DAUB-R_retina --device 0
python -m unittest discover -s tests
```

Use `python pred.py ...` for single-image inference and `python export.py ...` or `python video_onnx.py ...` for ONNX export and runtime checks.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, module-level constants in `UPPER_SNAKE_CASE`, functions and variables in `snake_case`, and clear `argparse` option names. Prefer small helper functions over inline path logic. Match the current style in touched files; no formatter configuration is committed, so avoid unrelated reformatting.

## Testing Guidelines
Add tests under `tests/` using `unittest`, following the current `test_*.py` pattern such as `tests/test_video_onnx.py`. Name test classes by feature and test methods by behavior, for example `test_parse_args_accepts_explicit_onnx_approx_mode`. Run `python -m unittest discover -s tests` before submitting changes. Add focused tests for new CLI flags, path resolution, and export or inference helpers.

## Commit & Pull Request Guidelines
Recent history uses concise Conventional Commit prefixes like `feat(...)`, `feat:`, and `chore:`. Keep messages imperative and scoped, for example `feat(export): add stream ONNX option`. PRs should explain the user-visible change, list validation commands, link related issues or paper tasks, and include sample output paths or screenshots when inference visuals change.

## Data & Artifact Handling
Do not commit datasets, large checkpoints, cached files, or `runs/` outputs. Keep machine-specific paths out of code and prefer `--dataset-root` or relative repository paths.
