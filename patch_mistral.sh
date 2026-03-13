#!/usr/bin/env bash
# Replaces the installed transformers modeling_mistral.py with the custom one.
# Mirrors the behavior in pytorch.Dockerfile.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/modeling_mistral.py"

TRANSFORMERS_PATH=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
DEST="$TRANSFORMERS_PATH/models/mistral/modeling_mistral.py"

echo "Copying $SRC -> $DEST"
cp "$SRC" "$DEST"
echo "Done."
