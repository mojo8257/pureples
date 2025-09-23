#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Start"

# 0) 安装 git-lfs（很多仓库需要，避免 push 报错）
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y git-lfs
  git lfs install || true
fi

# 1) Python venv
if [ ! -d ".venv" ]; then
  echo "[setup] create venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) 拉取子模块（容器重建时确保拿到）
git submodule update --init --recursive || true

# 3) 安装主项目（可编辑）
if [ -f "pyproject.toml" ] || [ -f "setup.cfg" ] || [ -f "setup.py" ]; then
  echo "[setup] pip install -e ."
  pip install -e .
fi

# 4) 安装子模块 neat-python（可编辑）
if [ -d "external/neat-python" ]; then
  if [ -f external/neat-python/pyproject.toml ] || [ -f external/neat-python/setup.cfg ] || [ -f external/neat-python/setup.py ]; then
    echo "[setup] pip install -e external/neat-python"
    pip install -e external/neat-python
  elif [ -f external/neat-python/requirements.txt ]; then
    pip install -r external/neat-python/requirements.txt
  fi
fi

# 5) 常用开发工具（可按需删）
pip install -U pytest black ruff isort

echo "[setup] Done"
python -V
pip -V
