# Kaggle Bootstrap Troubleshooting

If the first notebook cell prints:

```text
REPO_ROOT = /kaggle/working
SRC_ROOT exists = False
Kaggle input exists = True
```

then the Kaggle notebook is running without the repository source code. Replace
the first setup cell with this code, turn Kaggle notebook Internet on, and rerun
from the top.

```python
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

REPO_URL = 'https://github.com/joyjeni/context-pruning.git'
REPO_BRANCH = 'cursor/gemma-acpa-trust-safety-44f2'
REPO_ROOT = Path('/kaggle/working/context-pruning')
SRC_ROOT = REPO_ROOT / 'src'

if not (SRC_ROOT / 'acpa_gemma').exists():
    if REPO_ROOT.exists():
        print('Removing stale repo directory:', REPO_ROOT)
        shutil.rmtree(REPO_ROOT)
    print('Cloning repository branch:', REPO_BRANCH)
    try:
        subprocess.run(
            [
                'git',
                'clone',
                '--depth',
                '1',
                '--branch',
                REPO_BRANCH,
                REPO_URL,
                str(REPO_ROOT),
            ],
            check=True,
        )
    except Exception as exc:
        raise RuntimeError(
            'Could not clone the repository. Turn Kaggle Notebook Internet ON, '
            'then rerun from the first cell. If Internet is disabled, upload the '
            'repo as a Kaggle dataset and set REPO_ROOT to that dataset path.'
        ) from exc

if not (SRC_ROOT / 'acpa_gemma').exists():
    raise RuntimeError(f"acpa_gemma source package not found under {SRC_ROOT}")

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

print("REPO_ROOT =", REPO_ROOT)
print("SRC_ROOT exists =", SRC_ROOT.exists())
print("acpa_gemma exists =", (SRC_ROOT / 'acpa_gemma').exists())
print("Kaggle input exists =", Path("/kaggle/input").exists())

import acpa_gemma
print("Imported acpa_gemma from:", acpa_gemma.__file__)
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
