import json
import os
import subprocess
from pathlib import Path

import pandas as pd


TOON_CLI_PACKAGE = os.environ.get("TOON_CLI_PACKAGE", "@toon-format/cli")
TOON_NPM_CACHE = os.environ.get("TOON_NPM_CACHE", "/tmp/hyperagents-toon-npm-cache")


def _run_toon_cli(args, input_text):
    env = os.environ.copy()
    env.setdefault("npm_config_cache", TOON_NPM_CACHE)
    result = subprocess.run(
        ["npx", "--yes", TOON_CLI_PACKAGE, *args],
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "TOON CLI failed with exit code "
            f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def encode_toon(data):
    payload = json.dumps(data, ensure_ascii=False)
    return _run_toon_cli(["--encode"], payload)


def write_toon(df, path):
    path = Path(path)
    serializable = df.astype(object).where(pd.notna(df), None)
    toon_text = encode_toon(serializable.to_dict(orient="records"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toon_text, encoding="utf-8")


def read_toon(path):
    path = Path(path)
    payload = _run_toon_cli(["--decode"], path.read_text(encoding="utf-8"))
    records = json.loads(payload)
    if not isinstance(records, list):
        raise ValueError(f"Expected TOON dataset at {path} to decode to a list")
    return pd.DataFrame(records)
