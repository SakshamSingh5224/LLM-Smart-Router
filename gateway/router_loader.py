"""Load the Phase 2 router artifact; fall back to the static rules router if
it's missing or fails to load, per the plan's circuit-breaker/fallback requirement.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from smartrouter.classifier_router import ClassifierRouter, RulesRouter

log = logging.getLogger("smartrouter.router_loader")


def load_router(artifact_path: Union[str, Path], min_confidence: float = 0.0) -> tuple:
    """Returns (router, status_dict)."""
    path = Path(artifact_path)
    if not path.exists():
        log.warning("Router artifact not found at %s - using static rules fallback. "
                    "Run `make train-router` to train one.", path)
        return RulesRouter(), {"status": "degraded", "reason": "artifact_missing", "path": str(path)}
    try:
        router = ClassifierRouter.load(path, min_confidence=min_confidence)
        return router, {
            "status": "ok",
            "backend": router.artifact["backend"],
            "trained_at": router.artifact["trained_at"],
            "val_auc": router.artifact["val_auc"],
            "modes": {k: round(v, 4) for k, v in router.modes.items()},
            "default_mode": router.default_mode,
            "path": str(path),
        }
    except Exception as e:  # noqa: BLE001
        log.exception("Failed to load router artifact at %s", path)
        return RulesRouter(), {"status": "degraded", "reason": f"load_error: {e}", "path": str(path)}
