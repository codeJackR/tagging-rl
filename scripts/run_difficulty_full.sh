#!/usr/bin/env bash

# Launch the locked-SFT full difficulty pass with auditable operational logs.
# Run this only on the provisioned Vast.ai host from any directory in the repo.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EXPECTED_SCORER_COMMIT="69f1af14e3bbf0de6aa23c426e1190605f392802"
RL_PYTHON="/venv/rl/bin/python"
HF_CACHE_ROOT="/workspace/.hf_home"
OUTPUT_DIR="runs/sft-difficulty-k8"
SCORED_OUTPUT="data/train_weak_sft_scored.jsonl"
RUN_LOG="runs/sft-difficulty-k8-full.log"
MONITOR_LOG="runs/sft-difficulty-k8-monitor.csv"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--preflight-only" ) ]]; then
  echo "usage: $0 [--preflight-only]" >&2
  exit 2
fi
preflight_only=false
if [[ "${1:-}" == "--preflight-only" ]]; then
  preflight_only=true
fi

fail() {
  echo "preflight failed: $*" >&2
  exit 1
}

[[ -x "${RL_PYTHON}" ]] || fail "missing Python interpreter: ${RL_PYTHON}"
[[ -d "${HF_CACHE_ROOT}" ]] || fail "missing Hugging Face cache: ${HF_CACHE_ROOT}"
git merge-base --is-ancestor "${EXPECTED_SCORER_COMMIT}" HEAD \
  || fail "scorer commit is not an ancestor of HEAD"
git diff --quiet "${EXPECTED_SCORER_COMMIT}" -- \
  training/score_difficulty.py tests/test_score_difficulty.py \
  || fail "difficulty scorer or its tests changed after the reviewed commit"
git diff --quiet || fail "tracked worktree has unstaged changes"
git diff --cached --quiet || fail "tracked worktree has staged changes"

for path in "${OUTPUT_DIR}" "${SCORED_OUTPUT}" "${RUN_LOG}" "${MONITOR_LOG}"; do
  [[ ! -e "${path}" ]] || fail "refusing existing path: ${path}"
done

if [[ "${preflight_only}" == true ]]; then
  echo "launcher preflight passed; generation was not started"
  echo "git commit: $(git rev-parse HEAD)"
  echo "reviewed scorer commit: ${EXPECTED_SCORER_COMMIT}"
  echo "expected command: HF_HOME=${HF_CACHE_ROOT} PYTHONPATH=. ${RL_PYTHON} -m training.score_difficulty --full"
  exit 0
fi

started_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "full difficulty run start: ${started_at_utc}"
echo "git commit: $(git rev-parse HEAD)"
echo "reviewed scorer commit: ${EXPECTED_SCORER_COMMIT}"
echo "command: HF_HOME=${HF_CACHE_ROOT} PYTHONPATH=. ${RL_PYTHON} -m training.score_difficulty --full"
df -h /workspace
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader

echo "timestamp_utc,gpu_memory_used_mib,gpu_memory_free_mib,gpu_utilization_percent,gpu_temperature_c,disk_available_bytes" > "${MONITOR_LOG}"
(
  while true; do
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    gpu_sample="$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    disk_available="$(df --output=avail -B1 /workspace | tail -n 1 | tr -d ' ')"
    echo "${timestamp},${gpu_sample},${disk_available}"
    sleep 1
  done
) >> "${MONITOR_LOG}" &
monitor_pid=$!

stop_monitor() {
  if [[ -n "${monitor_pid:-}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  monitor_pid=""
}

trap stop_monitor EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_status=0
HF_HOME="${HF_CACHE_ROOT}" PYTHONPATH=. "${RL_PYTHON}" \
  -m training.score_difficulty --full || run_status=$?

stop_monitor
trap - EXIT INT TERM

ended_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "full difficulty run end: ${ended_at_utc}"
echo "scorer exit code: ${run_status}"

if [[ -s "${MONITOR_LOG}" ]]; then
  awk -F, '
    NR > 1 {
      if (samples == 0 || $2 > peak_gpu) peak_gpu = $2
      if (samples == 0 || $6 < minimum_disk) minimum_disk = $6
      samples += 1
    }
    END {
      printf "monitor samples: %d\n", samples
      if (samples > 0) {
        printf "peak GPU memory used: %d MiB\n", peak_gpu
        printf "minimum disk available: %d bytes\n", minimum_disk
      }
    }
  ' "${MONITOR_LOG}"
fi

for path in "${OUTPUT_DIR}/rollouts.jsonl.gz" "${SCORED_OUTPUT}" "${OUTPUT_DIR}/manifest.json"; do
  if [[ -f "${path}" ]]; then
    sha256sum "${path}"
  else
    echo "artifact absent: ${path}"
  fi
done

df -h /workspace
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader

exit "${run_status}"
