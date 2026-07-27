""""Frozen in git" is only real if drift is detectable.

Committing a file makes an *intentional* edit visible in review. It does nothing
about the edit nobody reviews: a re-run of the labeler that overwrites the file, a
stray import that reorders rows, a notebook that rewrites one cell. Those land as
a diff nobody reads, and every number computed before the change silently stops
being comparable to every number computed after.

So the freeze writes a checksum sidecar and `verify()` recomputes it. The eval
harness should call `verify()` before it reports anything.

The checksum is over the *canonical* serialization (sorted keys, fixed
separators), so re-serializing the same data on a different machine or Python
version produces the same digest. Key order is not signal.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .records import Row, canonical_line, read_jsonl, write_jsonl

SIDECAR_SUFFIX = ".frozen.json"


def checksum(rows: list[Row]) -> str:
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r.sku_id):
        h.update(canonical_line(row).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def sidecar_path(path: str | Path) -> Path:
    return Path(str(path) + SIDECAR_SUFFIX)


def freeze(rows: list[Row], path: str | Path, *, note: str = "") -> dict:
    """Write the split and its checksum sidecar. Commit both; tag the commit."""
    path = Path(path)
    write_jsonl(rows, path)
    digest = checksum(rows)
    meta = {
        "sha256": digest,
        "n_rows": len(rows),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corrected_rows": sum(1 for r in rows if r.provenance.human_corrected),
        "with_frontier_snapshot": sum(
            1 for r in rows if r.provenance.frontier_labels is not None
        ),
        "note": note,
    }
    sidecar_path(path).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def verify(path: str | Path) -> dict:
    """Recompute and compare. Call this before reporting any eval number."""
    path = Path(path)
    side = sidecar_path(path)
    if not side.exists():
        return {"ok": False, "reason": f"no checksum sidecar at {side}"}
    if not path.exists():
        return {"ok": False, "reason": f"missing dataset at {path}"}

    expected = json.loads(side.read_text())
    rows = read_jsonl(path)
    actual = checksum(rows)
    if actual == expected["sha256"]:
        return {"ok": True, "sha256": actual, "n_rows": len(rows)}
    return {
        "ok": False,
        "reason": "CHECKSUM MISMATCH — the frozen eval set has been modified",
        "expected": expected["sha256"],
        "actual": actual,
        "expected_rows": expected["n_rows"],
        "actual_rows": len(rows),
        "frozen_at": expected.get("frozen_at"),
        "consequence": (
            "Every metric recorded before this change is no longer comparable to "
            "anything measured after it. Restore from the tagged commit, or re-freeze "
            "deliberately and re-baseline."
        ),
    }
