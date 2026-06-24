import subprocess
import shutil
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).parent / "scripts" / "new-paper.sh"


def test_rejects_path_traversal():
    result = subprocess.run(
        ["bash", str(SCRIPT), "../evil"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_rejects_invalid_chars():
    for bad in ["MyPaper", "my paper", "my_paper", "paper!"]:
        result = subprocess.run(
            ["bash", str(SCRIPT), bad],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, f"should have rejected: {bad!r}"


def test_creates_data_dir():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "scripts"
        scripts_dir.mkdir()
        shutil.copy(SCRIPT, scripts_dir / "new-paper.sh")

        subprocess.run(
            ["bash", str(scripts_dir / "new-paper.sh"), "test-paper"],
            capture_output=True,
            cwd=tmp,
        )
        assert (Path(tmp) / "test-paper" / "data").is_dir()
