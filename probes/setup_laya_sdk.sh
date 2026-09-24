#!/usr/bin/env bash
# Validate Laya Multilingual runs locally on this box (RTX 3090) and expose real numbers.
set -x
WINPY="G:/dev/AI/laya/.venv/Scripts/python.exe"
cd /g/dev/AI/laya || exit 1

uv venv .venv --python 3.12 || exit 1
uv pip install --python "$WINPY" torch --torch-backend auto || exit 1
uv pip install --python "$WINPY" laya "laya[serve]" || exit 1
"$WINPY" -I -c "import laya, torch; print('laya', laya.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"