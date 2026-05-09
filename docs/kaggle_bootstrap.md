# Kaggle Bootstrap Troubleshooting

If the first notebook cell prints:

```text
REPO_ROOT = /kaggle/working
SRC_ROOT exists = False
Kaggle input exists = True
```

then the Kaggle notebook is running without the repository source code. Replace
the first setup cell with this code and rerun it.

```python
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/joyjeni/context-pruning.git"
REPO_BRANCH = "cursor/gemma-acpa-trust-safety-44f2"
CLONE_DIR = Path("/kaggle/working/context-pruning")

possible_roots = [
    Path.cwd(),
    CLONE_DIR,
    Path("/kaggle/input/context-pruning"),
]

REPO_ROOT = next((path for path in possible_roots if (path / "src/acpa_gemma").exists()), None)

if REPO_ROOT is None:
    print("Repository source not found. Cloning into", CLONE_DIR)
    if CLONE_DIR.exists() and CLONE_DIR.name == "context-pruning":
        print("Removing stale clone directory:", CLONE_DIR)
        shutil.rmtree(CLONE_DIR)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                REPO_BRANCH,
                REPO_URL,
                str(CLONE_DIR),
            ],
            check=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not clone the repository. In Kaggle, turn Internet ON in "
            "Notebook settings, then rerun this cell. If Internet is disabled, "
            "upload the repository as a Kaggle dataset and set REPO_ROOT to that path."
        ) from exc
    REPO_ROOT = CLONE_DIR

SRC_ROOT = REPO_ROOT / "src"
if not (SRC_ROOT / "acpa_gemma").exists():
    raise RuntimeError(f"acpa_gemma source package not found under {SRC_ROOT}")

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

print("REPO_ROOT =", REPO_ROOT)
print("SRC_ROOT exists =", SRC_ROOT.exists())
print("acpa_gemma exists =", (SRC_ROOT / "acpa_gemma").exists())
print("Kaggle input exists =", Path("/kaggle/input").exists())
```

Expected output:

```text
REPO_ROOT = /kaggle/working/context-pruning
SRC_ROOT exists = True
acpa_gemma exists = True
Kaggle input exists = True
```

## Fix `pip install` SyntaxError

Do not run this as Python:

```python
pip install -q google-genai pandas pyarrow tomli
```

That syntax is invalid in a Python code cell. Use one of these options.

Notebook shell magic:

```python
!pip install -q google-genai pandas pyarrow tomli
```

or valid Python:

```python
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "google-genai",
    "pandas",
    "pyarrow",
    "tomli",
])
```
