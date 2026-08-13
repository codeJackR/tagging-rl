"""Pure, metadata-only selector for the Run 2 confirmation set.

The selector deliberately has no file, network, labeling, or model-inference
code.  A later acquisition adapter may supply raw Shopify rows, but membership
is decided here using only product identity and pre-label metadata.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from training.run2_confirmation_contract import (
    MINIMUM_CLEAN_CANDIDATES,
    SELECTION_SEED,
    TARGET_ROWS,
)
from training.split_sft import group_key


VERSION = "grpo-run2-confirmation-selector-v1"
DEFAULT_MAX_ROWS_PER_FAMILY = 4
DEFAULT_MAX_ROWS_PER_STORE = 60
DEFAULT_MINIMUM_STORES = 8
UNMATCHED_CATEGORY = "<unmatched>"

# These are post-treatment fields: accepting any of them would make it possible
# to choose confirmation examples after seeing labels, difficulty, or outputs.
_FORBIDDEN_KEYS = frozenset(
    {
        "difficulty",
        "frontier_labels",
        "human_corrected",
        "labels",
        "model_output",
        "prediction",
        "predictions",
        "provenance",
        "reward",
        "rewards",
        "sft_pass_rate",
        "split",
    }
)
_WORD_BREAKS = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class SelectionPolicy:
    target_rows: int = TARGET_ROWS
    minimum_clean_candidates: int = MINIMUM_CLEAN_CANDIDATES
    seed: int = SELECTION_SEED
    maximum_rows_per_family: int = DEFAULT_MAX_ROWS_PER_FAMILY
    maximum_rows_per_store: int = DEFAULT_MAX_ROWS_PER_STORE
    minimum_stores: int = DEFAULT_MINIMUM_STORES

    def validate(self) -> None:
        if self.target_rows <= 0:
            raise ValueError("target_rows must be positive")
        if self.minimum_clean_candidates < self.target_rows:
            raise ValueError("minimum_clean_candidates must cover target_rows")
        if self.maximum_rows_per_family <= 0:
            raise ValueError("maximum_rows_per_family must be positive")
        if self.maximum_rows_per_store <= 0:
            raise ValueError("maximum_rows_per_store must be positive")
        if self.minimum_stores <= 0:
            raise ValueError("minimum_stores must be positive")
        if self.minimum_stores * self.maximum_rows_per_store < self.target_rows:
            raise ValueError("store constraints cannot possibly fill target_rows")


@dataclass(frozen=True)
class _Candidate:
    sku_id: str
    store: str
    title: str
    brand: str | None
    family_key: str
    provisional_category: str
    rank_sha256: str

    @property
    def stratum(self) -> tuple[str, str]:
        return (self.store, self.provisional_category)


def _stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _normalized_words(value: str) -> str:
    return _WORD_BREAKS.sub(" ", value.casefold()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def provisional_category(
    *,
    product_category: str | None,
    title: str,
    aliases: Mapping[str, str],
    category_order: Sequence[str],
) -> str:
    """Resolve a non-gold stratum from product type first, then title.

    Longer aliases win within one source.  The pack's canonical value order is
    the final deterministic tie-break.  No label or model output is accepted.
    """

    order = {value: index for index, value in enumerate(category_order)}
    unknown_canonicals = set(aliases.values()) - set(order)
    if unknown_canonicals:
        raise ValueError(
            "category aliases reference values absent from category_order: "
            f"{sorted(unknown_canonicals)}"
        )

    normalized_aliases: list[tuple[str, str]] = []
    for alias, canonical in aliases.items():
        phrase = _normalized_words(alias)
        if phrase:
            normalized_aliases.append((phrase, canonical))

    for raw_source in (product_category or "", title):
        text = _normalized_words(raw_source)
        matches = {
            (alias, canonical)
            for alias, canonical in normalized_aliases
            if _contains_phrase(text, alias)
        }
        if matches:
            alias, canonical = min(
                matches,
                key=lambda item: (-len(item[0]), order[item[1]], item[0]),
            )
            del alias
            return canonical
    return UNMATCHED_CATEGORY


def _find_forbidden_key(value: Any, path: str = "candidate") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text.casefold() in _FORBIDDEN_KEYS:
                return child_path
            found = _find_forbidden_key(child, child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_forbidden_key(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    stripped = value.strip()
    return stripped or None


def _parse_candidate(
    row: Mapping[str, Any],
    *,
    aliases: Mapping[str, str],
    category_order: Sequence[str],
    seed: int,
) -> _Candidate:
    forbidden = _find_forbidden_key(row)
    if forbidden is not None:
        raise ValueError(f"post-treatment field is forbidden before selection: {forbidden}")

    sku_id = _required_text(row.get("sku_id"), "candidate.sku_id")
    source = _required_text(row.get("source"), f"candidate {sku_id} source")
    if not source.startswith("shopify:") or source == "shopify:":
        raise ValueError(f"candidate {sku_id} source must be shopify:<domain>")
    store = source.removeprefix("shopify:")
    if not sku_id.startswith(f"shopify:{store}:"):
        raise ValueError(f"candidate {sku_id} does not match source {source}")

    raw_input = row.get("input")
    if not isinstance(raw_input, Mapping):
        raise ValueError(f"candidate {sku_id} input must be an object")
    title = _required_text(raw_input.get("title"), f"candidate {sku_id} input.title")
    brand = _optional_text(raw_input.get("brand"), f"candidate {sku_id} input.brand")
    product_category = _optional_text(
        raw_input.get("category"), f"candidate {sku_id} input.category"
    )
    family = group_key(
        SimpleNamespace(input=SimpleNamespace(brand=brand, title=title))
    )
    return _Candidate(
        sku_id=sku_id,
        store=store,
        title=title,
        brand=brand,
        family_key=family,
        provisional_category=provisional_category(
            product_category=product_category,
            title=title,
            aliases=aliases,
            category_order=category_order,
        ),
        rank_sha256=_stable_hash(seed, sku_id),
    )


def select_confirmation_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    prior_sku_ids: set[str] | frozenset[str],
    prior_family_keys: set[str] | frozenset[str],
    category_aliases: Mapping[str, str],
    category_order: Sequence[str],
    policy: SelectionPolicy = SelectionPolicy(),
) -> dict[str, Any]:
    """Exclude old identities, then select broad deterministic strata.

    The round-robin gives every store/category stratum one opportunity per pass.
    Candidate order inside each stratum is the contract's seeded SHA-256 order.
    """

    policy.validate()
    parsed: list[_Candidate] = []
    seen_skus: set[str] = set()
    exclusions: list[dict[str, str]] = []
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise TypeError("every candidate must be an object")
        candidate = _parse_candidate(
            raw,
            aliases=category_aliases,
            category_order=category_order,
            seed=policy.seed,
        )
        if candidate.sku_id in seen_skus:
            raise ValueError(f"duplicate candidate SKU: {candidate.sku_id}")
        seen_skus.add(candidate.sku_id)
        if candidate.sku_id in prior_sku_ids:
            exclusions.append(
                {"sku_id": candidate.sku_id, "reason": "prior_exact_sku_overlap"}
            )
        elif candidate.family_key in prior_family_keys:
            exclusions.append(
                {"sku_id": candidate.sku_id, "reason": "prior_family_overlap"}
            )
        else:
            parsed.append(candidate)

    if len(parsed) < policy.minimum_clean_candidates:
        raise RuntimeError(
            "family-clean candidate buffer below predeclared minimum: "
            f"{len(parsed)} < {policy.minimum_clean_candidates}"
        )
    clean_stores = {candidate.store for candidate in parsed}
    if len(clean_stores) < policy.minimum_stores:
        raise RuntimeError(
            "family-clean candidate buffer has too few stores: "
            f"{len(clean_stores)} < {policy.minimum_stores}"
        )

    queues: dict[tuple[str, str], deque[_Candidate]] = {}
    grouped: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
    for candidate in parsed:
        grouped[candidate.stratum].append(candidate)
    for stratum, members in grouped.items():
        queues[stratum] = deque(sorted(members, key=lambda item: item.rank_sha256))
    stratum_order = sorted(
        queues,
        key=lambda item: _stable_hash(policy.seed, f"stratum\0{item[0]}\0{item[1]}"),
    )

    selected: list[_Candidate] = []
    selected_skus: set[str] = set()
    family_counts: Counter[str] = Counter()
    store_counts: Counter[str] = Counter()
    while len(selected) < policy.target_rows:
        progress = False
        for stratum in stratum_order:
            queue = queues[stratum]
            while queue:
                candidate = queue.popleft()
                if family_counts[candidate.family_key] >= policy.maximum_rows_per_family:
                    continue
                if store_counts[candidate.store] >= policy.maximum_rows_per_store:
                    continue
                selected.append(candidate)
                selected_skus.add(candidate.sku_id)
                family_counts[candidate.family_key] += 1
                store_counts[candidate.store] += 1
                progress = True
                break
            if len(selected) == policy.target_rows:
                break
        if not progress:
            raise RuntimeError(
                "selection constraints exhausted candidates before target: "
                f"{len(selected)} < {policy.target_rows}"
            )

    if len(store_counts) < policy.minimum_stores:
        raise RuntimeError(
            f"selected only {len(store_counts)} stores; minimum is {policy.minimum_stores}"
        )

    unselected: list[dict[str, str]] = []
    for candidate in sorted(parsed, key=lambda item: item.rank_sha256):
        if candidate.sku_id in selected_skus:
            continue
        if family_counts[candidate.family_key] >= policy.maximum_rows_per_family:
            reason = "family_cap_reached"
        elif store_counts[candidate.store] >= policy.maximum_rows_per_store:
            reason = "store_cap_reached"
        else:
            reason = "target_filled"
        unselected.append({"sku_id": candidate.sku_id, "reason": reason})

    selected_rows = [
        {
            "selection_order": index,
            "sku_id": candidate.sku_id,
            "store": candidate.store,
            "family_key": candidate.family_key.replace("\0", "::"),
            "provisional_category": candidate.provisional_category,
            "stratum": [candidate.store, candidate.provisional_category],
            "rank_sha256": candidate.rank_sha256,
        }
        for index, candidate in enumerate(selected, start=1)
    ]
    category_counts = Counter(item.provisional_category for item in selected)
    return {
        "version": VERSION,
        "status": "confirmation_membership_selected_before_labeling",
        "policy": {
            "target_rows": policy.target_rows,
            "minimum_clean_candidates": policy.minimum_clean_candidates,
            "seed": policy.seed,
            "maximum_rows_per_family": policy.maximum_rows_per_family,
            "maximum_rows_per_store": policy.maximum_rows_per_store,
            "minimum_stores": policy.minimum_stores,
        },
        "selection_fields_used": [
            "sku_id",
            "source",
            "input.brand",
            "input.category",
            "input.title",
        ],
        "counts": {
            "candidate_rows": len(candidates),
            "prior_exact_sku_exclusions": sum(
                item["reason"] == "prior_exact_sku_overlap" for item in exclusions
            ),
            "prior_family_exclusions": sum(
                item["reason"] == "prior_family_overlap" for item in exclusions
            ),
            "family_clean_candidates": len(parsed),
            "selected_rows": len(selected),
            "selected_stores": len(store_counts),
            "selected_families": len(family_counts),
        },
        "selected_store_counts": dict(sorted(store_counts.items())),
        "selected_provisional_category_counts": dict(sorted(category_counts.items())),
        "selected": selected_rows,
        "excluded_before_selection": sorted(exclusions, key=lambda item: item["sku_id"]),
        "unselected_clean_candidates": unselected,
        "invariants": {
            "prior_exact_sku_overlap": 0,
            "prior_family_overlap": 0,
            "maximum_observed_rows_per_family": max(family_counts.values()),
            "maximum_observed_rows_per_store": max(store_counts.values()),
            "membership_uses_labels_or_model_outputs": False,
        },
        "execution_boundary": {
            "network_requests_performed": False,
            "frontier_labeling_performed": False,
            "model_inference_performed": False,
            "gpu_training_authorized": False,
        },
    }
