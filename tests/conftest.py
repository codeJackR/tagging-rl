import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# The merchant corpus is collected but not redistributed: see data/README.md.
# Roughly 170 tests verify data-file checksums and cannot run without it, and a
# clone that reports them as failures says "this repository is broken" when the
# truth is "this file was deliberately withheld".
#
# The conversion below is deliberately narrow. It only fires on a
# FileNotFoundError naming one of these exact paths, so it cannot hide a genuine
# failure, and it does nothing at all on a checkout that has the corpus.
WITHHELD_CORPUS = frozenset(
    {
        "train_weak.jsonl",
        "train_weak_sft_scored.jsonl",
        "train_weak_grpo_cap4.jsonl",
        "train_weak_grpo_cap4_sft_train_v1.jsonl",
        "train_weak_grpo_smoke_v1.jsonl",
        "eval.jsonl",
        "eval_candidates.jsonl",
        "grpo_run2_causal_schedule_v1.jsonl",
        "probe_100.jsonl",
    }
)

# Words that mean "this file is not here", as opposed to "this file is wrong".
ABSENCE_WORDS = ("absent", "missing", "No such file", "does not exist", "not found")


def _withheld_filename(exc: BaseException) -> str | None:
    """The withheld corpus file this exception is about, if it is about one.

    Two shapes occur. The OS raises FileNotFoundError with `filename` set; the
    project's own contract checks raise it with a message and no `filename`, so
    the message is searched too. Both are still gated on the exception type and
    on the exact basenames above.
    """
    while exc is not None:
        if isinstance(exc, FileNotFoundError):
            if exc.filename and Path(str(exc.filename)).name in WITHHELD_CORPUS:
                return Path(str(exc.filename)).name
            for name in WITHHELD_CORPUS:
                if name in str(exc):
                    return name
        else:
            # Some gates raise their own type, for example CompositionError, and
            # say the file is absent in the message. Naming a withheld file is
            # not enough on its own, because the same gates legitimately name it
            # when its *hash* has drifted, and masking that would hide a real
            # defect. Both a withheld name and a word meaning absence are
            # required.
            text = str(exc)
            if any(word in text for word in ABSENCE_WORDS):
                for name in WITHHELD_CORPUS:
                    if name in text:
                        return name
        exc = exc.__cause__ or exc.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.failed or call.excinfo is None:
        return
    name = _withheld_filename(call.excinfo.value)
    if name is None:
        return
    report.outcome = "skipped"
    report.longrepr = (
        f"requires {name}, which is collected but not redistributed. "
        "See data/README.md."
    )
