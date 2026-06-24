#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <paper-slug>"
  echo "Example: $0 attention-is-all-you-need"
  exit 1
fi

SLUG="$1"
DIR="$(cd "$(dirname "$0")/.." && pwd)/$SLUG"

if [[ -d "$DIR" ]]; then
  echo "Error: $SLUG already exists"
  exit 1
fi

mkdir -p "$DIR"

cat > "$DIR/PAPER.md" << 'EOF'
# <Title>

- **Authors:** ...
- **Year:** ...
- **Venue:** ...
- **Link:** https://arxiv.org/abs/...
- **Key contributions:**
  - ...
- **Implementation scope:** ...
EOF

cat > "$DIR/RESULTS.md" << 'EOF'
# Results

| Metric | Paper | Reproduced |
|--------|-------|------------|
| ...    | ...   | ...        |

## Notes

- Hardware: ...
- Training time: ...
- Hyperparams: ...
- Known deviations from paper: ...
EOF

cat > "$DIR/pyproject.toml" << EOF
[project]
name = "$SLUG"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
EOF

touch "$DIR/__init__.py"

cat > "$DIR/test_placeholder.py" << EOF
def test_placeholder():
    pass
EOF

echo "Created $SLUG/"
echo "  PAPER.md       <- fill in paper metadata"
echo "  RESULTS.md     <- fill in after running experiments"
echo "  pyproject.toml <- add dependencies"
echo "  __init__.py"
echo "  test_placeholder.py"
