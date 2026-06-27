# Papers-From-Scratch

Implementing ML/AI research papers from scratch in Python.

## Papers

| Slug | Title | Year | Status |
|------|-------|------|--------|
| [vision-transformer](vision-transformer/) | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 2020 | In Progress |

## Structure

Each paper lives in its own directory:

```
<paper-slug>/
├── PAPER.md              # title, authors, link, key contributions
├── RESULTS.md            # reproduced vs paper metrics
├── pyproject.toml        # per-paper dependencies
├── conftest.py
├── __init__.py
├── Code/
└── Tests/
```

**Naming:** kebab-case. Add year suffix when names collide (`resnet-2015` vs `resnet-v2-2016`).

## Adding a Paper

```bash
bash scripts/new-paper.sh <paper-slug>
```

Then:
1. Fill in `PAPER.md` with metadata
2. Add deps to `<slug>/pyproject.toml`
3. Write tests and implement following TDD cycle

## Running Tests

```bash
uv run pytest                         # all papers
uv run pytest <paper-slug>/Tests/      # one paper
uv run pytest -k "test_name"          # single test
```
