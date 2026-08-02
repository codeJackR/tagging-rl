"""Deterministic scoring primitives for the SFT difficulty pass.

Model loading and rollout generation are deliberately absent for now. Keeping
those concerns separate lets the correctness definition and derived-data write be
tested locally before any GPU time is spent.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from labeling.records import read_jsonl, write_jsonl
from training.dataset import SYSTEM
from training.predict import prompt_messages
from verifier import Pack, load_pack, verify

ROLLOUTS_PER_PROMPT = 8
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
DEFAULT_CHECKPOINT = "runs/sft-combined-2epoch/checkpoint-406"
DEFAULT_SOURCE = "data/train_weak.jsonl"
DEFAULT_PACK = "packs/vastraa_taste_v1"
DEFAULT_SELECTION_MANIFEST = "runs/sft-selection.json"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_NEW_TOKENS = 170
DEFAULT_SEED = 20260802
DEFAULT_BATCH_SIZE = 32
DEFAULT_GPU_MEMORY_UTILIZATION = 0.65
SMOKE_PRODUCTS = 2


def build_argument_parser() -> argparse.ArgumentParser:
    """Define an explicit, difficult-to-trigger-accidentally run contract."""
    parser = argparse.ArgumentParser(
        description="Measure per-product difficulty with eight locked-SFT rollouts."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help=f"score only the first {SMOKE_PRODUCTS} products",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="score all products; generates 28,800 completions for the default data",
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--pack", default=DEFAULT_PACK)
    parser.add_argument("--selection-manifest", default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--scored-output",
        help="derived dataset path; defaults by smoke/full mode",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="optional smaller smoke subset; forbidden with --full",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate mode-dependent options without importing GPU libraries."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.full and args.limit is not None:
        parser.error("--limit is allowed only with --smoke")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be in (0, 1)")

    if args.smoke:
        args.limit = args.limit or SMOKE_PRODUCTS
        args.output_dir = args.output_dir or "runs/sft-difficulty-k8-smoke"
        args.scored_output = args.scored_output or str(
            Path(args.output_dir) / "scored-smoke.jsonl"
        )
    else:
        args.limit = None
        args.output_dir = args.output_dir or "runs/sft-difficulty-k8"
        args.scored_output = args.scored_output or "data/train_weak_sft_scored.jsonl"

    try:
        _validate_generation_config(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            }
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return args


@dataclass(frozen=True)
class CompletionScore:
    """The auditable result of grading one raw model completion."""

    passed: bool
    schema_valid: bool
    vocab_valid: bool
    rule_violations: list[str]
    errors: list[str]
    correct_labels: int
    scorable_labels: int
    incorrect_labels: list[str]
    excluded_gold_unknown_labels: list[str]


@dataclass(frozen=True)
class RolloutRecord:
    """One generated answer plus enough evidence to explain its grade."""

    sku_id: str
    rollout_index: int
    raw_output: str
    passed: bool
    schema_valid: bool
    vocab_valid: bool
    rule_violations: list[str]
    errors: list[str]
    correct_labels: int
    scorable_labels: int
    incorrect_labels: list[str]
    excluded_gold_unknown_labels: list[str]
    generation_seed: int
    completion_tokens: int

    def __post_init__(self) -> None:
        if not self.sku_id:
            raise ValueError("rollout record requires a sku_id")
        if not 0 <= self.rollout_index < ROLLOUTS_PER_PROMPT:
            raise ValueError(
                f"rollout_index must be between 0 and {ROLLOUTS_PER_PROMPT - 1}"
            )
        if self.completion_tokens < 0:
            raise ValueError("completion_tokens cannot be negative")


def _is_unknown_gold(value, *, multi: bool, unknown_token: str) -> bool:
    """Gold abstention means no truth exists, so that field cannot be scored."""
    if multi:
        return value == [unknown_token]
    return value == unknown_token


def _labels_match(gold, predicted) -> bool:
    """Multi-value fields are sets semantically; their JSON order is irrelevant."""
    if isinstance(gold, list) and isinstance(predicted, list):
        return set(gold) == set(predicted)
    return gold == predicted


def score_completion(raw_output: str, gold: dict, pack: Pack) -> CompletionScore:
    """Grade one rollout using the locked strict-pass definition.

    A completion passes only when it is verifier-clean and every field with known
    gold matches exactly. Gold abstentions are excluded rather than treated as an
    answer key, because the listing supplied no truth for those fields.
    """
    verification = verify(raw_output, pack)
    parsed = verification.parsed
    correct = 0
    scorable = 0
    incorrect_labels: list[str] = []
    excluded_gold_unknown_labels: list[str] = []

    for name, spec in pack.specs.items():
        gold_value = gold.get(name, pack.unknown_token)
        if _is_unknown_gold(
            gold_value,
            multi=spec.kind == "multi",
            unknown_token=pack.unknown_token,
        ):
            excluded_gold_unknown_labels.append(name)
            continue
        scorable += 1
        if parsed is not None and _labels_match(gold_value, parsed.get(name)):
            correct += 1
        else:
            incorrect_labels.append(name)

    passed = (
        verification.ok
        and scorable > 0
        and correct == scorable
    )
    return CompletionScore(
        passed=passed,
        schema_valid=verification.schema_valid,
        vocab_valid=verification.vocab_valid,
        rule_violations=list(verification.rule_violations),
        errors=list(verification.errors),
        correct_labels=correct,
        scorable_labels=scorable,
        incorrect_labels=incorrect_labels,
        excluded_gold_unknown_labels=excluded_gold_unknown_labels,
    )


def calculate_pass_rate(results: Sequence[bool]) -> float:
    """Convert exactly eight binary rollout outcomes into a difficulty score."""
    if len(results) != ROLLOUTS_PER_PROMPT:
        raise ValueError(
            f"difficulty scoring requires exactly {ROLLOUTS_PER_PROMPT} rollouts"
        )
    if not all(isinstance(result, bool) for result in results):
        raise TypeError("rollout outcomes must be booleans")
    return sum(results) / ROLLOUTS_PER_PROMPT


def build_rollout_record(
    *,
    sku_id: str,
    rollout_index: int,
    raw_output: str,
    score: CompletionScore,
    generation_seed: int,
    completion_tokens: int,
) -> RolloutRecord:
    """Attach generation identity and raw evidence to a deterministic grade."""
    return RolloutRecord(
        sku_id=sku_id,
        rollout_index=rollout_index,
        raw_output=raw_output,
        passed=score.passed,
        schema_valid=score.schema_valid,
        vocab_valid=score.vocab_valid,
        rule_violations=list(score.rule_violations),
        errors=list(score.errors),
        correct_labels=score.correct_labels,
        scorable_labels=score.scorable_labels,
        incorrect_labels=list(score.incorrect_labels),
        excluded_gold_unknown_labels=list(score.excluded_gold_unknown_labels),
        generation_seed=generation_seed,
        completion_tokens=completion_tokens,
    )


def generate_rollouts(
    rows,
    *,
    pack: Pack,
    llm,
    tokenizer,
    sampling_params,
    lora_request,
    batch_size: int,
    generation_seed: int,
    progress=None,
) -> list[RolloutRecord]:
    """Generate, preserve, and immediately grade eight completions per product.

    `llm`, tokenizer, and vLLM request objects are injected so this control flow
    can be tested without importing CUDA libraries or loading model weights.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = list(rows)
    records: list[RolloutRecord] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                prompt_messages(row),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        request_outputs = llm.generate(
            prompts,
            sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        if len(request_outputs) != len(batch):
            raise RuntimeError(
                "vLLM returned a different number of requests than prompts: "
                f"{len(request_outputs)} != {len(batch)}"
            )

        for row, request_output in zip(batch, request_outputs, strict=True):
            completions = sorted(request_output.outputs, key=lambda output: output.index)
            indices = [output.index for output in completions]
            if indices != list(range(ROLLOUTS_PER_PROMPT)):
                raise RuntimeError(
                    f"{row.sku_id}: expected completion indices 0 through 7, "
                    f"received {indices}"
                )
            gold = row.to_verifier_record(pack)
            for output in completions:
                raw = output.text
                score = score_completion(raw, gold, pack)
                records.append(
                    build_rollout_record(
                        sku_id=row.sku_id,
                        rollout_index=output.index,
                        raw_output=raw,
                        score=score,
                        generation_seed=generation_seed,
                        completion_tokens=len(output.token_ids),
                    )
                )

        if progress is not None:
            progress(min(start + len(batch), len(rows)), len(rows))

    _validate_rollout_groups(records)
    return records


def create_vllm_backend(args):
    """Load the locked base model, LoRA adapter, and stochastic sampling config."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.model,  # Cached Qwen2.5 base weights.
        runner="generate",  # This run needs text generation, not embeddings.
        dtype="bfloat16",  # Native, stable precision on the RTX 3090.
        seed=args.seed,  # Reproduce engine-side scheduling and sampling state.
        max_model_len=896,  # Covers measured prompt + 170 completion tokens.
        gpu_memory_utilization=args.gpu_memory_utilization,  # Leave host-app headroom.
        enable_lora=True,  # Apply the locked SFT adapter without merging weights.
        max_lora_rank=16,  # Exactly matches the selected checkpoint's LoRA rank.
    )
    sampling_params = SamplingParams(
        n=ROLLOUTS_PER_PROMPT,  # Eight attempts define one product's pass rate.
        temperature=args.temperature,  # Introduce variation needed to measure difficulty.
        top_p=args.top_p,  # Trim the very low-probability tail while still sampling.
        max_tokens=args.max_new_tokens,  # The measured completion ceiling is below 170.
        seed=args.seed,  # Make this sampled baseline reproducible.
    )
    lora_request = LoRARequest(
        lora_name="locked-sft-checkpoint-406",  # Stable adapter identity in vLLM logs.
        lora_int_id=1,  # Positive process-local ID required by vLLM.
        lora_path=str(Path(args.adapter).resolve()),  # PEFT checkpoint directory.
        base_model_name=args.model,  # Explicitly bind the adapter to Qwen's base.
    )
    return llm, llm.get_tokenizer(), sampling_params, lora_request


def verify_locked_inputs(args) -> dict:
    """Prove that model, adapter, and source are the preselected SFT baseline."""
    selection_path = Path(args.selection_manifest)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "locked_before_frozen_eval":
        raise RuntimeError("SFT selection manifest is not in the locked state")

    selected = selection["selected_checkpoint"]
    if selected["base_model"] != args.model:
        raise RuntimeError(
            f"base model disagrees with lock: {args.model} != {selected['base_model']}"
        )
    if Path(selected["remote_path"]).resolve() != Path(args.adapter).resolve():
        raise RuntimeError(
            f"adapter path disagrees with lock: {args.adapter} != {selected['remote_path']}"
        )

    adapter_file = Path(args.adapter) / selected["adapter_weights"]["file"]
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    actual_adapter_sha = _sha256_file(adapter_file)
    expected_adapter_sha = selected["adapter_weights"]["sha256"]
    if actual_adapter_sha != expected_adapter_sha:
        raise RuntimeError(
            "locked adapter checksum mismatch: "
            f"{actual_adapter_sha} != {expected_adapter_sha}"
        )
    expected_bytes = selected["adapter_weights"].get("bytes")
    if expected_bytes is not None and adapter_file.stat().st_size != expected_bytes:
        raise RuntimeError("locked adapter byte size mismatch")

    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_source_sha = _sha256_file(source)
    expected_source_sha = selection["training_data"]["source_sha256"]
    if actual_source_sha != expected_source_sha:
        raise RuntimeError(
            "difficulty source disagrees with the locked SFT data: "
            f"{actual_source_sha} != {expected_source_sha}"
        )

    return {
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256_file(selection_path),
        "status": selection["status"],
        "adapter_sha256_verified": actual_adapter_sha,
        "source_sha256_verified": actual_source_sha,
    }


def _refuse_output_collisions(paths: Sequence[Path], *, overwrite: bool) -> None:
    collisions = [path for path in paths if path.exists()]
    if collisions and not overwrite:
        rendered = ", ".join(str(path) for path in collisions)
        raise FileExistsError(
            f"output already exists: {rendered}; pass --overwrite to replace it"
        )


def _pass_rates_from_records(records: Sequence[RolloutRecord]) -> dict[str, float]:
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        outcomes[record.sku_id].append(record.passed)
    return {
        sku_id: calculate_pass_rate(results)
        for sku_id, results in outcomes.items()
    }


def _git_state() -> dict:
    """Best-effort code provenance; artifact creation must not depend on Git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            check=False,
        ).returncode != 0
        return {"git_commit": commit, "tracked_worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "tracked_worktree_dirty": None}


def run_difficulty(args, *, backend_factory=create_vllm_backend) -> dict:
    """Execute one guarded smoke/full run and return its completed manifest."""
    output_dir = Path(args.output_dir)
    rollout_path = output_dir / "rollouts.jsonl.gz"
    manifest_path = output_dir / "manifest.json"
    scored_path = Path(args.scored_output)
    smoke_source_path = output_dir / "source-smoke.jsonl"
    collision_paths = [rollout_path, manifest_path, scored_path]
    if args.smoke:
        collision_paths.append(smoke_source_path)
    _refuse_output_collisions(collision_paths, overwrite=args.overwrite)

    # Verify all locked inputs before initializing CUDA or reserving vLLM memory.
    lock = verify_locked_inputs(args)
    pack = load_pack(args.pack)
    original_rows = read_jsonl(args.source)
    rows = original_rows[: args.limit] if args.limit is not None else original_rows
    if not rows:
        raise RuntimeError("difficulty source selection is empty")

    run_source_path = smoke_source_path if args.smoke else Path(args.source)

    llm, tokenizer, sampling_params, lora_request = backend_factory(args)
    records = generate_rollouts(
        rows,
        pack=pack,
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        lora_request=lora_request,
        batch_size=args.batch_size,
        generation_seed=args.seed,
        progress=lambda done, total: print(f"generated {done}/{total} products"),
    )
    pass_rates = _pass_rates_from_records(records)
    if args.smoke:
        write_jsonl(rows, smoke_source_path)
    write_rollout_records(records, rollout_path)
    write_scored_dataset(run_source_path, scored_path, pass_rates)

    generation_config = {
        "backend": "vllm",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "structured_outputs": False,
    }
    manifest = build_difficulty_manifest(
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_path=run_source_path,
        scored_path=scored_path,
        rollout_path=rollout_path,
        checkpoint_path=args.adapter,
        adapter_path=Path(args.adapter) / "adapter_model.safetensors",
        base_model=args.model,
        pack=pack,
        prompt_text=SYSTEM,
        pass_rates=pass_rates,
        generation_config=generation_config,
    )
    manifest["checkpoint_lock"] = lock
    manifest["code"] = _git_state()
    manifest["mode"] = "smoke" if args.smoke else "full"
    write_difficulty_manifest(manifest, manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_cli_args(argv)
    manifest = run_difficulty(args)
    summary = manifest["summary"]
    print(
        "difficulty scoring complete: "
        f"products={summary['products_scored']}, "
        f"rollouts={summary['total_rollouts']}, "
        f"retained={summary['retained_for_grpo']}, "
        f"manifest={Path(args.output_dir) / 'manifest.json'}"
    )
    return 0


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _validate_rollout_groups(records: Sequence[RolloutRecord]) -> None:
    if not records:
        raise ValueError("rollout artifact cannot be empty")

    by_sku: dict[str, set[int]] = defaultdict(set)
    for record in records:
        key = (record.sku_id, record.rollout_index)
        if record.rollout_index in by_sku[record.sku_id]:
            raise ValueError(f"duplicate rollout record for {key}")
        by_sku[record.sku_id].add(record.rollout_index)

    expected = set(range(ROLLOUTS_PER_PROMPT))
    incomplete = sorted(sku for sku, indices in by_sku.items() if indices != expected)
    if incomplete:
        preview = ", ".join(incomplete[:3])
        raise ValueError(
            "every SKU must have rollout indices 0 through 7 exactly once; "
            f"incomplete: {preview}"
        )


def write_rollout_records(
    records: Sequence[RolloutRecord], path: str | Path
) -> int:
    """Write canonical, deterministically compressed rollout evidence.

    Sorting by SKU/index makes output independent of batching order. A zero gzip
    timestamp and empty embedded filename make identical records byte-identical
    across runs and output paths, so the artifact hash is meaningful.
    """
    records = list(records)
    _validate_rollout_groups(records)
    path = Path(path)
    if not str(path).endswith(".jsonl.gz"):
        raise ValueError("rollout artifact path must end in .jsonl.gz")
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(records, key=lambda record: (record.sku_id, record.rollout_index))
    payload = ("\n".join(_canonical_json(asdict(record)) for record in ordered) + "\n").encode(
        "utf-8"
    )
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(payload)
    return len(ordered)


def write_scored_dataset(
    source_path: str | Path,
    output_path: str | Path,
    pass_rates: dict[str, float],
) -> int:
    """Write a complete derived dataset while preserving the source byte-for-byte."""
    source = Path(source_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("refusing to overwrite the source difficulty dataset")

    rows = read_jsonl(source)
    source_ids = [row.sku_id for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source dataset contains duplicate sku_id values")

    missing = set(source_ids) - set(pass_rates)
    extra = set(pass_rates) - set(source_ids)
    if missing or extra:
        raise ValueError(
            "pass-rate SKU set must exactly match the source dataset "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    scored_rows = []
    for row in rows:
        rate = pass_rates[row.sku_id]
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"invalid pass rate for {row.sku_id}: {rate}")
        scored = row.model_copy(deep=True)
        scored.difficulty.sft_pass_rate = rate
        scored_rows.append(scored)

    return write_jsonl(scored_rows, output)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rollout_records(path: str | Path) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(RolloutRecord(**json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid rollout record at line {line_number}: {exc}"
                ) from exc
    _validate_rollout_groups(records)
    return records


def _validate_created_at(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("created_at_utc must include a UTC offset")


def _validate_generation_config(config: dict) -> None:
    required = {"temperature", "top_p", "max_new_tokens", "seed"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"generation config missing: {', '.join(sorted(missing))}")
    if config["temperature"] <= 0:
        raise ValueError("difficulty rollouts require temperature > 0")
    if not 0 < config["top_p"] <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if config["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not isinstance(config["seed"], int):
        raise ValueError("generation seed must be an integer")


def build_difficulty_manifest(
    *,
    created_at_utc: str,
    source_path: str | Path,
    scored_path: str | Path,
    rollout_path: str | Path,
    checkpoint_path: str | Path,
    adapter_path: str | Path,
    base_model: str,
    pack: Pack,
    prompt_text: str,
    pass_rates: dict[str, float],
    generation_config: dict,
) -> dict:
    """Cross-check all difficulty artifacts and build their provenance manifest."""
    _validate_created_at(created_at_utc)
    _validate_generation_config(generation_config)

    source_path = Path(source_path)
    scored_path = Path(scored_path)
    rollout_path = Path(rollout_path)
    adapter_path = Path(adapter_path)
    for named_path in (source_path, scored_path, rollout_path, adapter_path):
        if not named_path.is_file():
            raise FileNotFoundError(named_path)

    source_rows = read_jsonl(source_path)
    scored_rows = read_jsonl(scored_path)
    source_ids = {row.sku_id for row in source_rows}
    scored_by_sku = {row.sku_id: row for row in scored_rows}
    if len(source_ids) != len(source_rows):
        raise ValueError("source dataset contains duplicate sku_id values")
    if len(scored_by_sku) != len(scored_rows):
        raise ValueError("scored dataset contains duplicate sku_id values")
    if source_ids != set(scored_by_sku) or source_ids != set(pass_rates):
        raise ValueError("source, scored dataset, and pass-rate SKU sets disagree")

    for sku_id, expected_rate in pass_rates.items():
        if round(expected_rate * ROLLOUTS_PER_PROMPT) != expected_rate * ROLLOUTS_PER_PROMPT:
            raise ValueError(f"pass rate is not a multiple of 1/8 for {sku_id}")
        saved_rate = scored_by_sku[sku_id].difficulty.sft_pass_rate
        if saved_rate != expected_rate:
            raise ValueError(
                f"pass rates disagree for {sku_id}: expected {expected_rate}, saved {saved_rate}"
            )

    rollout_records = _read_rollout_records(rollout_path)
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for record in rollout_records:
        outcomes[record.sku_id].append(record.passed)
    if set(outcomes) != source_ids:
        raise ValueError("rollout and source dataset SKU sets disagree")
    for sku_id, results in outcomes.items():
        actual_rate = calculate_pass_rate(results)
        if actual_rate != pass_rates[sku_id]:
            raise ValueError(
                f"rollout and scored pass rates disagree for {sku_id}: "
                f"{actual_rate} != {pass_rates[sku_id]}"
            )

    histogram = Counter(f"{rate:.3f}" for rate in pass_rates.values())
    passed_rollouts = sum(record.passed for record in rollout_records)
    generation = dict(generation_config)
    generation["rollouts_per_product"] = ROLLOUTS_PER_PROMPT

    vocab_path = pack.path / "vocab.yaml"
    rules_path = pack.path / "rules.yaml"
    return {
        "version": "sft-difficulty-v1",
        "created_at_utc": created_at_utc,
        "model": {
            "base_model": base_model,
            "checkpoint": str(checkpoint_path),
            "adapter_file": str(adapter_path),
            "adapter_bytes": adapter_path.stat().st_size,
            "adapter_sha256": _sha256_file(adapter_path),
        },
        "data": {
            "source": str(source_path),
            "source_sha256": _sha256_file(source_path),
            "source_rows": len(source_rows),
            "scored": str(scored_path),
            "scored_sha256": _sha256_file(scored_path),
            "scored_rows": len(scored_rows),
        },
        "pack": {
            "name": pack.name,
            "path": str(pack.path),
            "vocab_sha256": _sha256_file(vocab_path),
            "rules_sha256": _sha256_file(rules_path),
        },
        "prompt": {
            "source": "training.dataset.SYSTEM",
            "sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        },
        "generation": generation,
        "scoring": {
            "pass_definition": (
                "verifier-clean and every scorable gold label matches exactly"
            ),
            "gold_unknown_policy": "excluded",
            "multi_value_comparison": "order-insensitive exact set match",
            "all_same_group_policy": "exclude pass rates 0.0 and 1.0 from GRPO",
        },
        "artifacts": {
            "rollouts": str(rollout_path),
            "rollouts_sha256": _sha256_file(rollout_path),
            "rollout_records": len(rollout_records),
        },
        "summary": {
            "products_scored": len(source_rows),
            "total_rollouts": len(rollout_records),
            "passed_rollouts": passed_rollouts,
            "failed_rollouts": len(rollout_records) - passed_rollouts,
            "retained_for_grpo": sum(0.0 < rate < 1.0 for rate in pass_rates.values()),
            "always_failed": sum(rate == 0.0 for rate in pass_rates.values()),
            "always_passed": sum(rate == 1.0 for rate in pass_rates.values()),
            "pass_rate_histogram": dict(sorted(histogram.items())),
        },
    }


def write_difficulty_manifest(manifest: dict, path: str | Path) -> None:
    """Write the human-readable manifest with stable key ordering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
