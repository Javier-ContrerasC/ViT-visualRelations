"""
Lightweight local stub to neutralize W&B calls when you want to run
the repository without wandb/network access.

This file intentionally provides a minimal, local-compatible API that
mirrors the parts of wandb used by this repo: `init`, `log`, `finish`,
`run.id`, `Settings`, `sweep`, and `agent`.

Placing this file in the repo root makes Python import this local module
instead of the real `wandb` package, so you don't need to modify other
files. To restore normal WandB behavior, remove or rename this file and
install/enable the real `wandb` package.
"""
import json
import os
import uuid
from types import SimpleNamespace

# Simple in-memory run object
class _Run:
    def __init__(self):
        self.id = str(uuid.uuid4())


_global_run = None


class Settings:
    def __init__(self, **kwargs):
        # store settings but do nothing with them
        self._settings = kwargs


def init(project=None, entity=None, dir=None, settings=None, config=None, **kwargs):
    """Create a fake run object. Mirrors `wandb.init` minimal behavior.

    This signature accepts `config` and arbitrary kwargs for compatibility with
    scripts that pass additional arguments to `wandb.init`.
    """
    global _global_run, run
    _global_run = _Run()
    run = SimpleNamespace(id=_global_run.id)
    print(f"[wandb stub] init called. project={project}, entity={entity}, dir={dir}, config={'<present>' if config is not None else None}")
    return run


def log(data):
    """Log metrics locally to stdout (non-blocking)."""
    try:
        s = json.dumps(data, default=str)
    except Exception:
        s = str(data)
    print(f"[wandb stub] log: {s}")


def finish():
    """Finish the fake run."""
    global _global_run
    if _global_run is not None:
        print(f"[wandb stub] finish run id={_global_run.id}")
    _global_run = None


def sweep(sweep=None, project=None, entity=None):
    """Store sweep config locally and return a fake sweep id.

    This avoids network calls when `sweep.py` runs. The sweep
    configuration (if provided) is written to `.local_wandb_sweep.json`.
    """
    sid = f"local-sweep-{uuid.uuid4()}"
    try:
        if sweep is not None:
            with open(".local_wandb_sweep.json", "w", encoding="utf-8") as f:
                json.dump({"project": project, "entity": entity, "sweep": sweep}, f, default=str, indent=2)
            print(f"[wandb stub] sweep saved to .local_wandb_sweep.json (id={sid})")
        else:
            print(f"[wandb stub] sweep called with no config (id={sid})")
    except Exception as e:
        print(f"[wandb stub] sweep: failed to save sweep config: {e}")
    return sid


def agent(sweep_id, project=None, entity=None, count=1):
    """Agent stub: does not launch remote runs. Prints instructions.

    The real `wandb.agent` coordinates workers that pull configurations
    from WandB's backend. This stub simply informs the user how to run
    locally using the saved `.local_wandb_sweep.json` (if present).
    """
    print(f"[wandb stub] agent called. sweep_id={sweep_id}, project={project}, entity={entity}, count={count}")
    cfg_path = ".local_wandb_sweep.json"
    if os.path.exists(cfg_path):
        print(f"[wandb stub] Found local sweep config at {cfg_path}.")
        print("[wandb stub] To run locally, open that file and invoke train.py with the desired parameter combination.")
    else:
        print("[wandb stub] No local sweep config found. Nothing to run.")


# Minimal compatibility aliases
# Module-level `run` is set by `init()` above. Default to `None` before init.
run = None
