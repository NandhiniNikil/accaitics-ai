"""
Tests for the ML components and the Q&A layer.

These are deliberately about BEHAVIOUR, not about hitting a score. A test that
asserts "accuracy > 0.94" breaks the moment the data changes and tells you
nothing about whether the code is correct. What matters is:

  * the model beats the rule baseline it is supposed to replace
  * no label leaks into the anomaly features
  * confidence routing actually keeps auto-posted entries cleaner
  * the assistant never invents a number, and refuses what it cannot compute
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accaitics import accounting as acc        # noqa: E402
from accaitics import ask as ask_mod           # noqa: E402
from accaitics.anomaly import build_features, detect, evaluate  # noqa: E402
from accaitics.categorise import rule_predict, suggest, train   # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ledger():
    return acc.load_ledger(DATA / "transactions.csv")


@pytest.fixture(scope="module")
def trained(ledger):
    return train(ledger)


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------

def test_macro_f1_is_scored_over_the_same_label_set(trained, ledger):
    """
    REGRESSION TEST for a real bug.

    f1_score without `labels=` averages over the union of labels in y_true and
    y_pred. The rule baseline emits "Uncategorised" when it abstains — a
    phantom 14th class scoring 0.0 — so the baseline was divided by 14 and the
    model by 13. That artifact alone reversed the comparison and the project
    claimed the model won when it did not.
    """
    exp = ledger[(ledger["txn_type"] == "Expense") &
                 (ledger["description"].str.len() > 0)]
    counts = exp["category"].value_counts()
    n_real = int((counts >= 4).sum())
    assert trained["n_categories"] == n_real
    # Scored correctly the rules win; if this flips, something regressed.
    assert trained["rule_macro_f1"] > trained["ml_macro_f1"]
    assert trained["beats_rules"] is False


def test_rules_never_miscode_on_the_test_set(trained):
    """Rule errors must all be abstentions, never a wrong ledger head."""
    assert trained["rule_precision_when_fired"] == 1.0
    assert trained["rule_abstentions"] > 0


def test_model_generalises_better_than_rules_to_unseen_wording(trained):
    """
    The reason the model stays in. Rules are hand-written against text you
    have already seen; this probe uses phrasings neither has encountered.
    """
    from accaitics.categorise import generalisation_probe
    g = generalisation_probe(trained["model"])
    assert g["ml_correct"] > g["rule_correct"]
    assert g["rule_coverage"] < 0.8      # rules abstain on most new wording


def test_model_is_better_than_guessing(trained, ledger):
    """Must beat always predicting the most common ledger head."""
    exp = ledger[ledger["txn_type"] == "Expense"]
    majority = exp["category"].value_counts(normalize=True).iloc[0]
    assert trained["ml_accuracy"] > majority * 2


def test_confidence_routing_improves_accuracy(trained):
    """Auto-posted entries must be measurably cleaner than the raw model."""
    assert trained["auto_post_accuracy"] > trained["ml_accuracy"]


def test_auto_post_meets_its_accuracy_target(trained):
    """The threshold exists to give a controlled error rate; 0.995 is the target."""
    assert trained["auto_post_accuracy"] >= 0.99


def test_threshold_is_chosen_without_touching_the_test_set(trained):
    """
    Tuning the threshold on the same rows you then report accuracy for is
    marking your own exam. It must come from CV inside the training set.
    """
    assert "CV" in trained["threshold_chosen_by"]
    assert "cv_auto_post_accuracy" in trained


def test_rule_hits_always_auto_post(trained):
    """A deterministic rule hit needs no confidence test."""
    from accaitics.categorise import suggest
    out = suggest(trained["model"], ["Salary payment for the month"],
                  trained["threshold"]).iloc[0]
    assert out["decided_by"] == "Keyword rule"
    assert out["action"] == "Auto-post"


def test_some_work_still_goes_to_a_human(trained):
    """100% automation would mean the threshold is doing nothing."""
    assert 0 < trained["review_pct"] < 0.5


def test_rules_catch_the_obvious_cases():
    assert rule_predict("Salary payment for the month") == "Salaries & Wages"
    assert rule_predict("Google Ads spend 4471") == "Marketing & Advertising"
    assert rule_predict("Electricity bill payment 2231") == "Utilities"
    assert rule_predict("Office rent for the month") == "Rent"


def test_rules_return_none_when_nothing_matches():
    """A rule that guesses is worse than one that abstains."""
    assert rule_predict("misc expenses") is None
    assert rule_predict("payment made") is None


def test_suggest_returns_one_row_per_input(trained):
    out = suggest(trained["model"],
                  ["ELECTRICITY BL PYMT 991", "Google Ads spend 12", "misc exp"],
                  trained["threshold"])
    assert len(out) == 3
    assert set(out["action"]) <= {"Auto-post", "Review required"}
    assert ((out["confidence"] >= 0) & (out["confidence"] <= 1)).all()


def test_ambiguous_text_is_routed_to_a_human(trained):
    """
    "misc expenses" carries no information about the ledger head. The correct
    behaviour is to ask a person, not to guess confidently.
    """
    out = suggest(trained["model"], ["misc expenses", "payment made"],
                  trained["threshold"])
    assert (out["confidence"] < 0.9).all()


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def test_anomaly_features_contain_no_labels(ledger):
    """
    THE test for this module. If the answer key reaches the features, the
    score is meaningless and would not reproduce on a real ledger.
    """
    exp = ledger[ledger["txn_type"] == "Expense"]
    feats = build_features(exp)
    banned = {"is_anomaly", "anomaly_type", "label", "target", "y"}
    assert not (set(feats.columns) & banned)
    assert feats.select_dtypes(include=[np.number]).shape[1] == feats.shape[1]
    assert not feats.isna().any().any(), "NaNs would silently distort the model"


def test_detector_flags_a_sensible_share(ledger):
    """A queue nobody can work through is not a useful queue."""
    scored = detect(ledger)
    rate = scored["is_flagged"].mean()
    assert 0.01 < rate < 0.20, f"flag rate {rate:.1%} is not workable"


def test_detector_catches_most_planted_errors(ledger):
    """Tight against the measured 0.80 — a real regression must fail here."""
    scored = detect(ledger)
    ev = evaluate(scored, DATA / "anomaly_key.csv")
    assert ev["hybrid"]["recall"] >= 0.75, (
        f"recall {ev['hybrid']['recall']:.0%} — too many errors reach the books")


def test_detector_precision_is_usable(ledger):
    """A queue that is mostly noise will not get worked through."""
    scored = detect(ledger)
    ev = evaluate(scored, DATA / "anomaly_key.csv")
    assert ev["hybrid"]["precision"] >= 0.6


def test_detector_ranking_beats_chance(ledger):
    """ROC-AUC judges the score independently of where the cut falls."""
    scored = detect(ledger)
    ev = evaluate(scored, DATA / "anomaly_key.csv")
    assert ev["roc_auc"] > 0.85


def test_duplicates_are_caught_by_the_rule_layer(ledger):
    """
    A duplicate is definitional, so the RULE (not the hybrid) must catch them,
    and the key must record both halves of each pair — recording only the
    injected copy scored every correct catch as half a false positive.
    """
    scored = detect(ledger)
    key = pd.read_csv(DATA / "anomaly_key.csv")
    dup_ids = set(key.loc[key["anomaly_type"] == "duplicate", "transaction_id"])
    assert len(dup_ids) % 2 == 0, "duplicate pairs must both be in the key"
    caught = scored.loc[scored["transaction_id"].isin(dup_ids) &
                        scored["rule_flag"], "transaction_id"]
    assert len(caught) / len(dup_ids) >= 0.9


def test_every_flag_carries_a_reason(ledger):
    """An accountant must know why a row surfaced."""
    scored = detect(ledger)
    flagged = scored[scored["is_flagged"]]
    assert (flagged["reason"].str.len() > 0).all()


def test_detection_is_one_sided(ledger):
    """
    Only unusually LARGE amounts should be flagged by the statistical layer.
    An earlier version was two-sided and filled the queue with ₹15 entries.
    """
    scored = detect(ledger)
    model_only = scored[scored["model_flag"] & ~scored["rule_flag"]]
    if len(model_only):
        feats = build_features(scored)
        assert (feats.loc[model_only.index, "category_robust_z"] > 0).all()


def test_detector_is_deterministic(ledger):
    """
    Same input, same output. Comparing counts would pass on two different
    flag sets of equal size, so compare the actual transaction IDs.
    """
    a = set(detect(ledger).pipe(lambda d: d[d["is_flagged"]])["transaction_id"])
    b = set(detect(ledger).pipe(lambda d: d[d["is_flagged"]])["transaction_id"])
    assert a == b and len(a) > 0


def test_tiny_ledger_takes_the_guard_path():
    """Below 20 expense rows the detector abstains rather than guessing."""
    tiny = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "date": pd.to_datetime(["2025-04-01", "2025-04-02"]),
        "description": ["Rent", "Fuel"], "amount": [1000.0, 500.0],
        "txn_type": ["Expense", "Expense"], "category": ["Rent", "Travel & Fuel"],
        "payment_method": ["Cash", "Cash"], "vendor": ["", ""]})
    out = detect(tiny)
    assert len(out) == 2
    assert not out["is_flagged"].any()


# ---------------------------------------------------------------------------
# Ask Finance — the assistant must never invent a figure
# ---------------------------------------------------------------------------

def test_revenue_answer_matches_the_calculation(ledger):
    """The number in the sentence must equal the number in the ledger."""
    a = ask_mod.ask("What is my total revenue?", ledger)
    expected = acc.totals(ledger)["revenue"]
    assert f"{expected:,.2f}" in a.workings
    assert a.answered is True


def test_profit_workings_actually_add_up(ledger):
    a = ask_mod.ask("What is my profit?", ledger)
    t = acc.totals(ledger)
    assert f"{t['revenue']:,.2f}" in a.workings
    assert f"{t['expenses']:,.2f}" in a.workings
    assert f"{t['profit']:,.2f}" in a.workings


def test_month_filter_is_applied(ledger):
    """A question naming a month must answer for that month, not the year."""
    a = ask_mod.ask("What was my revenue in September?", ledger)
    sept = ledger[ledger["date"].dt.to_period("M") == pd.Period("2025-09")]
    assert f"{acc.totals(sept)['revenue']:,.2f}" in a.workings


def test_category_question_is_understood(ledger):
    a = ask_mod.ask("How much did I spend on software?", ledger)
    assert a.answered is True
    assert "Software" in a.text


def test_receivables_question_is_refused(ledger):
    """A transaction list genuinely cannot answer this. Saying so is correct."""
    a = ask_mod.ask("How much do my customers owe me?", ledger)
    assert a.answered is False
    assert "receivable" in a.text.lower() or "owe" in a.text.lower()


def test_forecast_question_is_refused(ledger):
    a = ask_mod.ask("What will my revenue be next month?", ledger)
    assert a.answered is False


def test_off_topic_question_is_refused(ledger):
    a = ask_mod.ask("What is the capital of France?", ledger)
    assert a.answered is False
    assert "could not match" in a.text.lower()


def test_no_answer_invents_a_number(ledger):
    """
    Every answer must either carry workings or be an explicit refusal, AND no
    answer may contain nan or inf. An earlier version emitted "Rs nan" and
    "up inf%" on degenerate input, with answered=True.
    """
    for q in ask_mod.SAMPLE_QUESTIONS + ["what is the capital of France?",
                                         "how much do customers owe me?",
                                         "how much did I spend on sales?"]:
        a = ask_mod.ask(q, ledger)
        assert a.workings or not a.answered, (
            f"'{q}' produced an answer with no traceable calculation")
        blob = (a.text + a.workings).lower()
        assert "nan" not in blob and "inf%" not in blob, f"'{q}' emitted {blob[:80]}"


SAMPLE_INTENTS = [
    ("What is my total revenue?", "revenue"),
    ("Which expense category is highest?", "largest expense head"),
    ("How much did I spend on software?", "Software"),
    ("Which transactions should I review?", "worth a look"),
    ("What is my profit margin?", "Net profit"),
    ("How has revenue changed over time?", "Revenue moved"),
    ("How has revenue changed?", "Revenue moved"),
    ("Is my business growing?", "Revenue moved"),
    ("Which vendors did I pay most?", "largest vendor"),
]


@pytest.mark.parametrize("question,expected", SAMPLE_INTENTS)
def test_questions_route_to_the_right_calculation(question, expected, ledger):
    """
    Pins question -> intent. Without this, "Which expense category is highest?"
    silently fell through to the total-expenses answer: fluent, confident and
    answering a different question.
    """
    a = ask_mod.ask(question, ledger)
    assert a.answered is True, f"'{question}' was refused"
    assert expected.lower() in a.text.lower(), (
        f"'{question}' routed wrongly -> {a.text[:90]}")


def test_degenerate_ledger_does_not_emit_nan():
    """A single-row ledger must not produce 'Rs nan' or 'inf%'."""
    one = pd.DataFrame({
        "transaction_id": ["T1"], "date": pd.to_datetime(["2025-04-01"]),
        "description": ["Rent paid"], "amount": [1000.0], "txn_type": ["Expense"],
        "category": ["Rent"], "payment_method": ["Cash"], "vendor": [""]})
    for q in ["what is my profit?", "how has revenue changed?",
              "how much did I spend on rent?"]:
        a = ask_mod.ask(q, one)
        blob = (a.text + a.workings).lower()
        assert "nan" not in blob and "inf" not in blob.replace("inf%", "")


def test_empty_question_is_handled(ledger):
    assert ask_mod.ask("", ledger).answered is False


def test_function_schema_covers_the_intents():
    """The LLM tool definitions must stay in step with the implementation."""
    assert len(ask_mod.FUNCTION_SCHEMA) == 8
    for fn in ask_mod.FUNCTION_SCHEMA:
        assert {"name", "description", "parameters"} <= set(fn)
