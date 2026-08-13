from pathlib import Path

import pytest

from training.run2_checkpoint_monitor_runtime import (
    _checkpoint_identity,
    _generation_kwargs,
    publish_monitor_bundle,
)


def _checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    return checkpoint


def test_checkpoint_identity_requires_peft_files(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    identity = _checkpoint_identity(checkpoint)
    assert identity["adapter_model_sha256"]
    (checkpoint / "adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="PEFT"):
        _checkpoint_identity(checkpoint)


def test_success_bundle_is_atomic_complete_and_collision_safe(tmp_path):
    output = tmp_path / "monitor-step-100"
    manifest = publish_monitor_bundle(
        output_dir=output,
        greedy=[{"sku_id": "one", "raw": "{}"}],
        sampled=[{"sku_id": "one", "raw": "{}", "repeat": 0, "seed": 1}],
        report={"status": "checkpoint_outputs_scored"},
        resource={"model_reference_released": True},
        manifest_context={"mode": "smoke", "quality_evidence": False},
    )
    assert manifest["status"] == "checkpoint_monitor_complete"
    assert {path.name for path in output.iterdir()} == {
        "greedy.jsonl",
        "sampled.jsonl",
        "report.json",
        "resource.json",
        "manifest.json",
    }
    with pytest.raises(FileExistsError, match="exists"):
        publish_monitor_bundle(
            output_dir=output,
            greedy=[],
            sampled=[],
            report={},
            resource={},
            manifest_context={},
        )


def test_confirmation_named_output_is_rejected_before_staging(tmp_path):
    with pytest.raises(ValueError, match="confirmation"):
        publish_monitor_bundle(
            output_dir=tmp_path / "confirmation" / "monitor",
            greedy=[],
            sampled=[],
            report={},
            resource={},
            manifest_context={},
        )
    assert not (tmp_path / "confirmation").exists()


def test_generation_parameters_are_explicit_for_both_decoding_modes():
    tokenizer = type("Tokenizer", (), {"pad_token_id": 4, "eos_token_id": 5})()
    greedy = _generation_kwargs(
        tokenizer=tokenizer,
        max_completion_length=170,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    assert greedy == {
        "max_new_tokens": 170,
        "do_sample": False,
        "pad_token_id": 4,
        "eos_token_id": 5,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
    }
    sampled = _generation_kwargs(
        tokenizer=tokenizer,
        max_completion_length=170,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )
    assert sampled["do_sample"] is True
    assert sampled["temperature"] == 0.7
    assert sampled["top_p"] == 0.95
    assert "top_k" not in sampled
