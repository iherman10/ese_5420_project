# Notebook sync with jupytext
Created: 2026-11-05
Scope: How `.py` files and Jupyter notebooks are paired and committed. Commands and rules are in `CLAUDE.md`.

- Pair every notebook with a percent-format `.py` file through `[tool.jupytext]` in `pyproject.toml`; edit the `.py` by default.
- Commit both the `.py` and the `.ipynb`, including outputs.
- Commit `.vscode/settings.json` and `.vscode/extensions.json` so the setup carries across computers.
- Execute with `--set-kernel python3` because jupytext found no kernel matching the venv's Python on its own, and this avoids registering one per machine.
- Execute with `--run-path .` because jupytext otherwise runs from the notebook's folder, while VS Code runs from the repo root.
