"""Behavioral contract for measuring SFT difficulty before GRPO.

These tests intentionally exercise only deterministic scoring and artifact writing.
Model loading and generation belong in the CLI integration path and are tested by
the small remote smoke run, not by the local unit suite.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from labeling.records import read_jsonl, write_jsonl
from training.score_difficulty import (
    CompletionScore,
    build_difficulty_manifest,
    build_rollout_record,
    calculate_pass_rate,
    create_vllm_backend,
    generate_rollouts,
    parse_cli_args,
    run_difficulty,
    score_completion,
    summarize_rollout_metrics,
    write_difficulty_manifest,
    write_rollout_records,
    write_scored_dataset,
)
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train_weak.jsonl"


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


@pytest.fixture(scope="module")
def rows():
    return read_jsonl(TRAIN)


def test_exact_valid_completion_passes_and_gold_unknowns_are_ignored(pack, rows):
    row = rows[0]
    gold = row.to_verifier_record(pack)
    prediction = dict(gold)

    # The first real row has unknown gold fields. A valid committed answer in one
    # of those fields must not be graded against our lack of ground truth.
    unknown_field = next(
        name for name, label in row.labels.items() if label.status.value == "unknown"
    )
    spec = pack.specs[unknown_field]
    prediction[unknown_field] = next(
        value for value in spec.values if value != pack.unknown_token
    )

    result = score_completion(json.dumps(prediction), gold, pack)

    assert result.passed
    assert result.schema_valid
    assert result.vocab_valid
    assert result.rule_violations == []
    assert result.correct_labels == result.scorable_labels
    assert unknown_field in result.excluded_gold_unknown_labels
    assert result.incorrect_labels == []
    assert unknown_field in result.missed_abstention_labels
    assert not result.unknown_aware_passed


def test_valid_but_wrong_label_does_not_pass(pack, rows):
    gold = rows[0].to_verifier_record(pack)
    prediction = dict(gold)
    prediction["material"] = next(
        value for value in pack.specs["material"].values if value != gold["material"]
    )

    result = score_completion(json.dumps(prediction), gold, pack)

    assert not result.passed
    assert result.schema_valid
    assert result.vocab_valid
    assert result.correct_labels == result.scorable_labels - 1
    assert result.incorrect_labels == ["material"]


def test_unknown_on_known_gold_is_a_false_abstention(pack, rows):
    gold = rows[0].to_verifier_record(pack)
    prediction = dict(gold)
    prediction["material"] = pack.unknown_token

    result = score_completion(json.dumps(prediction), gold, pack)

    assert not result.passed
    assert result.false_abstention_labels == ["material"]
    assert "material" in result.incorrect_labels


def test_semantic_and_abstention_metrics_are_reported_separately(pack, rows):
    row = rows[0]
    gold = row.to_verifier_record(pack)
    prediction = dict(gold)
    unknown_field = next(
        name for name, label in row.labels.items() if label.status.value == "unknown"
    )
    prediction[unknown_field] = next(
        value for value in pack.specs[unknown_field].values
        if value != pack.unknown_token
    )
    raw = json.dumps(prediction)
    score = score_completion(raw, gold, pack)
    records = [
        build_rollout_record(
            sku_id=row.sku_id,
            rollout_index=index,
            raw_output=raw,
            score=score,
            generation_seed=42,
            completion_tokens=10,
        )
        for index in range(8)
    ]

    metrics = summarize_rollout_metrics(records, [row], pack)

    assert metrics["semantic_known_gold"]["macro_f1"] == 1.0
    assert metrics["semantic_known_gold"]["cell_exact_accuracy"] == 1.0
    assert metrics["abstention"]["micro_precision"] == 1.0
    assert metrics["abstention"]["micro_recall"] < 1.0
    assert metrics["abstention"]["micro_f1"] < 1.0
    assert metrics["unknown_aware_pass"]["passed_rollouts"] == 0


def test_multi_value_label_order_does_not_matter(pack, rows):
    row = next(
        row
        for row in rows
        if isinstance(row.to_verifier_record(pack)["details"], list)
        and len(row.to_verifier_record(pack)["details"]) > 1
    )
    gold = row.to_verifier_record(pack)
    prediction = dict(gold)
    prediction["details"] = list(reversed(gold["details"]))

    assert score_completion(json.dumps(prediction), gold, pack).passed


def test_malformed_completion_fails_cleanly(pack, rows):
    result = score_completion("not json", rows[0].to_verifier_record(pack), pack)

    assert not result.passed
    assert not result.schema_valid
    assert not result.vocab_valid
    assert result.correct_labels == 0


def test_pass_rate_requires_exactly_eight_rollouts():
    assert calculate_pass_rate([True, True, True, True, True, False, False, False]) == 0.625

    with pytest.raises(ValueError, match="exactly 8"):
        calculate_pass_rate([True, False])


def test_scored_dataset_is_derived_without_overwriting_source(pack, rows, tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "scored.jsonl"
    source.write_text(rows[0].model_dump_json() + "\n", encoding="utf-8")
    original = source.read_bytes()

    write_scored_dataset(source, output, {rows[0].sku_id: 0.625})

    assert source.read_bytes() == original
    scored = read_jsonl(output)
    assert scored[0].difficulty.sft_pass_rate == 0.625

    with pytest.raises(ValueError, match="overwrite"):
        write_scored_dataset(source, source, {rows[0].sku_id: 0.625})


def fake_score(passed: bool, *, errors=()) -> CompletionScore:
    return CompletionScore(
        passed=passed,
        schema_valid=not errors,
        vocab_valid=not errors,
        rule_violations=[],
        errors=list(errors),
        correct_labels=10 if passed else 9,
        scorable_labels=10,
        incorrect_labels=[] if passed else ["material"],
        excluded_gold_unknown_labels=["colour_primary"],
        correct_abstention_labels=["colour_primary"],
        missed_abstention_labels=[],
        false_abstention_labels=[],
        unknown_aware_passed=passed,
    )


def rollout_group(sku_id: str, passed: int):
    return [
        build_rollout_record(
            sku_id=sku_id,
            rollout_index=index,
            raw_output=f'{{"sample":{index}}}',
            score=fake_score(index < passed),
            generation_seed=20260801,
            completion_tokens=20 + index,
        )
        for index in range(8)
    ]


def test_rollout_record_keeps_raw_evidence_and_failure_reason():
    score = fake_score(False, errors=["schema: not valid JSON"])
    record = build_rollout_record(
        sku_id="sku-1",
        rollout_index=3,
        raw_output="not json",
        score=score,
        generation_seed=42,
        completion_tokens=2,
    )

    assert record.sku_id == "sku-1"
    assert record.rollout_index == 3
    assert record.raw_output == "not json"
    assert record.errors == ["schema: not valid JSON"]
    assert record.incorrect_labels == ["material"]
    assert record.excluded_gold_unknown_labels == ["colour_primary"]
    assert record.correct_abstention_labels == ["colour_primary"]
    assert not record.passed


def test_rollout_gzip_is_canonical_and_requires_complete_groups(tmp_path):
    records = rollout_group("sku-b", 5) + rollout_group("sku-a", 8)
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    assert write_rollout_records(records, first) == 16
    assert write_rollout_records(list(reversed(records)), second) == 16
    assert first.read_bytes() == second.read_bytes()

    with gzip.open(first, "rt", encoding="utf-8") as handle:
        saved = [json.loads(line) for line in handle]
    assert saved[0]["sku_id"] == "sku-a"
    assert saved[0]["rollout_index"] == 0
    assert saved[-1]["sku_id"] == "sku-b"
    assert saved[-1]["rollout_index"] == 7

    with pytest.raises(ValueError, match="indices 0 through 7"):
        write_rollout_records(records[:-1], tmp_path / "incomplete.jsonl.gz")


def test_manifest_hashes_artifacts_and_summarizes_selection(pack, rows, tmp_path):
    source = tmp_path / "source.jsonl"
    scored = tmp_path / "scored.jsonl"
    rollouts = tmp_path / "rollouts.jsonl.gz"
    adapter = tmp_path / "adapter_model.safetensors"
    manifest_path = tmp_path / "manifest.json"
    sample_rows = rows[:2]
    write_jsonl(sample_rows, source)
    rates = {sample_rows[0].sku_id: 1.0, sample_rows[1].sku_id: 0.625}
    write_scored_dataset(source, scored, rates)
    write_rollout_records(
        rollout_group(sample_rows[0].sku_id, 8)
        + rollout_group(sample_rows[1].sku_id, 5),
        rollouts,
    )
    adapter.write_bytes(b"test adapter weights")

    manifest = build_difficulty_manifest(
        created_at_utc="2026-08-01T23:00:00Z",
        source_path=source,
        scored_path=scored,
        rollout_path=rollouts,
        checkpoint_path="runs/sft-combined-2epoch/checkpoint-406",
        adapter_path=adapter,
        base_model="unsloth/Qwen2.5-1.5B-Instruct",
        pack=pack,
        prompt_text="system prompt",
        pass_rates=rates,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 170,
            "seed": 20260801,
        },
    )
    write_difficulty_manifest(manifest, manifest_path)

    saved = json.loads(manifest_path.read_text())
    assert saved["version"] == "sft-difficulty-v2"
    assert saved["model"]["adapter_sha256"] == manifest["model"]["adapter_sha256"]
    assert saved["data"]["source_rows"] == 2
    assert saved["generation"]["rollouts_per_product"] == 8
    assert saved["summary"]["total_rollouts"] == 16
    assert saved["summary"]["passed_rollouts"] == 13
    assert saved["summary"]["retained_for_grpo"] == 1
    assert saved["summary"]["pass_rate_histogram"] == {"0.625": 1, "1.000": 1}
    assert len(saved["artifacts"]["rollouts_sha256"]) == 64
    assert len(saved["prompt"]["sha256"]) == 64
    assert saved["scoring"]["gold_unknown_policy"] == "excluded"
    assert "semantic_known_gold" in saved["metrics"]
    assert "abstention" in saved["metrics"]
    assert "unknown_aware_pass" in saved["metrics"]


def test_manifest_rejects_rates_that_disagree_with_scored_dataset(pack, rows, tmp_path):
    source = tmp_path / "source.jsonl"
    scored = tmp_path / "scored.jsonl"
    rollouts = tmp_path / "rollouts.jsonl.gz"
    adapter = tmp_path / "adapter_model.safetensors"
    write_jsonl(rows[:1], source)
    sku = rows[0].sku_id
    write_scored_dataset(source, scored, {sku: 1.0})
    write_rollout_records(rollout_group(sku, 8), rollouts)
    adapter.write_bytes(b"weights")

    with pytest.raises(ValueError, match="disagree"):
        build_difficulty_manifest(
            created_at_utc="2026-08-01T23:00:00Z",
            source_path=source,
            scored_path=scored,
            rollout_path=rollouts,
            checkpoint_path="checkpoint-406",
            adapter_path=adapter,
            base_model="qwen",
            pack=pack,
            prompt_text="prompt",
            pass_rates={sku: 0.625},
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_new_tokens": 170,
                "seed": 1,
            },
        )


def test_cli_smoke_defaults_to_two_products_and_locked_checkpoint():
    args = parse_cli_args(["--smoke"])

    assert args.limit == 2
    assert args.adapter == "runs/sft-combined-2epoch/checkpoint-406"
    assert args.model == "unsloth/Qwen2.5-1.5B-Instruct"
    assert args.output_dir == "runs/sft-difficulty-k8-smoke"
    assert args.temperature == 0.7
    assert args.top_p == 0.95
    assert args.max_new_tokens == 170


def test_cli_full_run_requires_explicit_mode_and_forbids_limit():
    assert parse_cli_args(["--full"]).limit is None

    with pytest.raises(SystemExit):
        parse_cli_args([])
    with pytest.raises(SystemExit):
        parse_cli_args(["--full", "--limit", "2"])


def test_cli_rejects_invalid_sampling_values():
    with pytest.raises(SystemExit):
        parse_cli_args(["--smoke", "--temperature", "0"])
    with pytest.raises(SystemExit):
        parse_cli_args(["--smoke", "--top-p", "1.1"])


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return json.dumps(messages)


class FakeLLM:
    def __init__(self, raw_outputs):
        self.raw_outputs = raw_outputs
        self.calls = []

    def generate(self, prompts, sampling_params, **kwargs):
        self.calls.append((prompts, sampling_params, kwargs))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(index=index, text=raw, token_ids=[1, 2, 3])
                    for index, raw in reversed(list(enumerate(outputs)))
                ]
            )
            for outputs in self.raw_outputs
        ]


def test_generation_loop_preserves_and_grades_all_eight_outputs(pack, rows):
    row = rows[0]
    gold = json.dumps(row.to_verifier_record(pack))
    llm = FakeLLM([[gold] * 8])
    progress = []

    records = generate_rollouts(
        [row],
        pack=pack,
        llm=llm,
        tokenizer=FakeTokenizer(),
        sampling_params="sampling-config",
        lora_request="adapter-request",
        batch_size=1,
        generation_seed=42,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert len(records) == 8
    assert [record.rollout_index for record in records] == list(range(8))
    assert all(record.raw_output == gold for record in records)
    assert all(record.passed for record in records)
    assert all(record.completion_tokens == 3 for record in records)
    assert progress == [(1, 1)]
    assert llm.calls[0][2] == {
        "use_tqdm": False,
        "lora_request": "adapter-request",
    }


def test_generation_loop_rejects_incomplete_vllm_group(pack, rows):
    gold = json.dumps(rows[0].to_verifier_record(pack))
    llm = FakeLLM([[gold] * 7])

    with pytest.raises(RuntimeError, match="indices 0 through 7"):
        generate_rollouts(
            rows[:1],
            pack=pack,
            llm=llm,
            tokenizer=FakeTokenizer(),
            sampling_params="sampling-config",
            lora_request="adapter-request",
            batch_size=1,
            generation_seed=42,
        )


def test_vllm_uses_supported_native_sampler_when_flashinfer_is_absent(
    monkeypatch, tmp_path
):
    class FakeVLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_tokenizer(self):
            return "tokenizer"

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLoRARequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    import sys

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeVLLM, SamplingParams=FakeSamplingParams),
    )
    monkeypatch.setitem(sys.modules, "vllm.lora", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.lora.request",
        SimpleNamespace(LoRARequest=FakeLoRARequest),
    )
    args = SimpleNamespace(
        model="qwen",
        adapter=tmp_path,
        seed=42,
        gpu_memory_utilization=0.65,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=170,
    )

    create_vllm_backend(args)

    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert args.sampler_backend == "native"


def test_guarded_smoke_orchestration_writes_cross_checked_artifacts(
    pack, rows, tmp_path
):
    source = tmp_path / "source.jsonl"
    adapter_dir = tmp_path / "checkpoint-406"
    adapter_dir.mkdir()
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_file.write_bytes(b"locked adapter")
    write_jsonl(rows[:2], source)

    selection = tmp_path / "sft-selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "locked_before_frozen_eval",
                "selected_checkpoint": {
                    "remote_path": str(adapter_dir),
                    "base_model": "test-qwen",
                    "adapter_weights": {
                        "file": adapter_file.name,
                        "bytes": adapter_file.stat().st_size,
                        "sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
                    },
                },
                "training_data": {
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    args = parse_cli_args(
        [
            "--smoke",
            "--model",
            "test-qwen",
            "--adapter",
            str(adapter_dir),
            "--source",
            str(source),
            "--pack",
            str(pack.path),
            "--selection-manifest",
            str(selection),
            "--output-dir",
            str(output_dir),
        ]
    )

    def fake_backend(_args):
        raw_groups = [
            [json.dumps(row.to_verifier_record(pack))] * 8 for row in rows[:2]
        ]
        return FakeLLM(raw_groups), FakeTokenizer(), "sampling", "adapter"

    manifest = run_difficulty(args, backend_factory=fake_backend)

    assert manifest["mode"] == "smoke"
    assert manifest["summary"]["products_scored"] == 2
    assert manifest["summary"]["total_rollouts"] == 16
    assert manifest["summary"]["always_passed"] == 2
    assert manifest["checkpoint_lock"]["status"] == "locked_before_frozen_eval"
    assert (output_dir / "source-smoke.jsonl").is_file()
    assert (output_dir / "rollouts.jsonl.gz").is_file()
    assert (output_dir / "scored-smoke.jsonl").is_file()
    assert (output_dir / "manifest.json").is_file()

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_difficulty(args, backend_factory=fake_backend)
