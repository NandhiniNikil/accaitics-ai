"""
Step 4 — Anomaly detection: business rules + Isolation Forest.

WHAT THIS IS AND IS NOT
-----------------------
This flags transactions that do not look like the rest, so that an accountant
can LOOK AT THEM. That is all it does.

A flag is not proof of an error and certainly not proof of fraud. Some flagged
entries are perfectly legitimate — a genuine one-off equipment purchase looks
exactly like a keying mistake to a statistical model. The output is a review
queue; the person working through it is the control, not the model.

WHY UNSUPERVISED
----------------
A real ledger has no column saying "this row is wrong". If you knew which rows
were wrong you would not need a detector. So Isolation Forest sees only the
engineered features, never the answer key. The key in data/anomaly_key.csv is
used ONLY at the end, to measure how the detector did.

WHY RULES AS WELL AS A MODEL
----------------------------
Some errors are definitional rather than statistical. The same vendor and the
same amount twice within a few days is a duplicate — that is what the word
means, and a rule catches it with total explainability and no model required.
Statistical detection earns its place on the fuzzy cases a rule cannot
describe.

Measured on this ledger the rules alone score marginally better on F1 than
rules plus Isolation Forest (see the output of this script). Isolation Forest
is kept anyway, deliberately: the rules only catch error types somebody
thought to write a rule for, and the next error in a real ledger will be one
nobody anticipated. That is precisely the gap unsupervised detection fills.
The cost of keeping it is three extra rows in the review queue.

THE FEATURE THAT MATTERS MOST
-----------------------------
Raw amount is a poor feature on its own: a large raw-material bill is normal,
a stationery bill of the same size is not. So the strongest feature is how
unusual an amount is FOR ITS OWN LEDGER HEAD, measured on a log scale (expense
amounts are heavily right-skewed) with robust statistics — median and MAD
rather than mean and standard deviation, so a handful of large errors cannot
inflate the very yardstick they are measured against.

And it is ONE-SIDED. Only unusually LARGE amounts are flagged. An overstated
expense understates profit and may be fraud; an unusually small expense is
rarely worth an accountant's time. An earlier version of this module was
two-sided and filled the review queue with ₹15 and ₹24 entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

CONTAMINATION = 0.06      # prior on how dirty a ledger is; sets the IF cut
Z_LARGE = 2.0             # robust z above which an amount is "far above normal"
Z_WEEKEND = 1.5           # a weekend posting must also be this large to matter


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn each expense row into numbers. Every feature is something an
    accountant would also look at by eye.
    """
    f = pd.DataFrame(index=df.index)
    amt = df["amount"].abs()

    f["log_amount"] = np.log1p(amt)
    f["is_negative"] = (df["amount"] < 0).astype(int)
    f["day_of_week"] = df["date"].dt.dayofweek
    f["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    f["day_of_month"] = df["date"].dt.day
    f["is_round_25000"] = ((amt > 0) & (amt % 25000 == 0)).astype(int)

    # How unusual is this amount for its own ledger head? (log scale, robust)
    logged = np.log1p(amt)
    grp = logged.groupby(df["category"])
    med = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median()).replace(0, np.nan)
    f["category_robust_z"] = (0.6745 * (logged - med) / mad).fillna(0).clip(-25, 25)

    # Same question against this vendor's own history
    vmed = logged.groupby(df["vendor"].fillna("")).transform("median")
    f["vendor_log_ratio"] = (logged - vmed).fillna(0).clip(-10, 10)

    # Does this exact vendor + amount pair appear more than once?
    key = df["vendor"].fillna("").astype(str) + "|" + amt.round(2).astype(str)
    f["same_vendor_amount_count"] = key.map(key.value_counts()).fillna(1)

    return f


# ---------------------------------------------------------------------------
# Rule layer — deterministic, explainable, high precision
# ---------------------------------------------------------------------------

def apply_rules(f: pd.DataFrame) -> dict[str, pd.Series]:
    """Each rule returns a boolean mask, with a plain-English name."""
    return {
        "Possible duplicate — same vendor and amount more than once":
            f["same_vendor_amount_count"] > 1,
        "Suspiciously round amount":
            f["is_round_25000"] == 1,
        "Negative expense — likely a sign error":
            f["is_negative"] == 1,
        "Large payment posted at a weekend":
            (f["is_weekend"] == 1) & (f["category_robust_z"] > Z_WEEKEND),
        f"Far above the usual range for its ledger head":
            f["category_robust_z"] > Z_LARGE,
    }


def detect(df: pd.DataFrame, contamination: float = CONTAMINATION,
           random_state: int = 42) -> pd.DataFrame:
    """
    Score every expense row. Returns the ledger with score, flags and reasons.

    No labels are used anywhere in this function.
    """
    exp = df[df["txn_type"] == "Expense"].copy()
    if len(exp) < 20:
        exp["anomaly_score"] = 0.0
        exp["rule_flag"] = False
        exp["model_flag"] = False
        exp["is_flagged"] = False
        exp["reason"] = ""
        exp["signals"] = 0
        return exp

    feats = build_features(exp)

    # --- rule layer ---
    rules = apply_rules(feats)
    rule_frame = pd.DataFrame(rules, index=exp.index)
    exp["rule_flag"] = rule_frame.any(axis=1)

    # --- model layer (Isolation Forest) ---
    X = StandardScaler().fit_transform(feats)
    model = IsolationForest(n_estimators=300, contamination=contamination,
                            random_state=random_state, n_jobs=-1)
    pred = model.fit_predict(X)
    exp["anomaly_score"] = -model.score_samples(X)        # higher = more unusual
    # One-sided: an unusually SMALL expense is not a finding worth an
    # accountant's time, so only above-median rows can be flagged.
    exp["model_flag"] = (pred == -1) & (feats["category_robust_z"] > 0)

    exp["is_flagged"] = exp["rule_flag"] | exp["model_flag"]

    # --- plain-English reasons ---
    reasons = []
    for i in exp.index:
        bits = [name for name, mask in rules.items() if bool(mask.loc[i])]
        if exp.loc[i, "model_flag"] and not bits:
            bits.append("Unusual combination of amount, timing and vendor")
        elif exp.loc[i, "model_flag"]:
            bits.append("Also flagged as statistically unusual")
        reasons.append("; ".join(bits))
    exp["reason"] = reasons
    exp["signals"] = rule_frame.sum(axis=1) + exp["model_flag"].astype(int)

    # Rank by how many independent signals fired, then by value at stake.
    # Sorting by amount alone just surfaces the largest legitimate purchases.
    return exp.sort_values(["signals", "amount"], ascending=[False, False])


def evaluate(scored: pd.DataFrame, key_path: Path) -> dict:
    """
    Score the detector against the planted-error key.

    IMPORTANT CAVEAT: this measures how well the detector finds errors WE
    PLANTED. It does not establish a fraud-detection accuracy. Real errors are
    not drawn from the same distribution as synthetic ones, and real fraud is
    designed to look normal.
    """
    key = pd.read_csv(key_path)
    truth = set(key["transaction_id"])
    y = scored["transaction_id"].isin(truth).astype(int).to_numpy()

    def prf(mask):
        p, r, f1, _ = precision_recall_fscore_support(
            y, mask.astype(int), average="binary", zero_division=0)
        return {"precision": float(p), "recall": float(r), "f1": float(f1),
                "n_flagged": int(mask.sum())}

    hybrid = prf(scored["is_flagged"].to_numpy())
    rules_only = prf(scored["rule_flag"].to_numpy())
    model_only = prf(scored["model_flag"].to_numpy())

    auc = float(roc_auc_score(y, scored["anomaly_score"])) if y.sum() else float("nan")

    caught = int((scored["is_flagged"].to_numpy() & (y == 1)).sum())
    by_type = (key.set_index("transaction_id")
               .join(scored.set_index("transaction_id")[["is_flagged"]], how="left")
               .fillna({"is_flagged": False})
               .groupby("anomaly_type")["is_flagged"]
               .agg(planted="size", caught="sum").reset_index())
    missed = scored[(~scored["is_flagged"]) & (y == 1)]

    return {
        "n_reviewed": int(len(scored)),
        "n_planted": int(y.sum()),
        "hybrid": hybrid, "rules_only": rules_only, "model_only": model_only,
        "roc_auc": auc,
        "flag_rate": float(scored["is_flagged"].mean()),
        "true_positives": caught,
        "false_positives": int(scored["is_flagged"].sum()) - caught,
        "missed": int(len(missed)),
        "by_type": by_type,
        "missed_rows": missed,
        "truth": truth,
    }


def main() -> None:
    from accaitics.accounting import load_ledger

    print("=" * 70)
    print("STEP 4 — ANOMALY DETECTION (RULES + ISOLATION FOREST)")
    print("=" * 70)

    df = load_ledger(ROOT / "data" / "transactions.csv")
    scored = detect(df)
    ev = evaluate(scored, ROOT / "data" / "anomaly_key.csv")

    print(f"\nReviewed {ev['n_reviewed']} expense entries.")
    print(f"Contamination prior {CONTAMINATION:.0%} — an ASSUMPTION about how dirty")
    print("the ledger is, which directly sets how many rows Isolation Forest flags.")

    print(f"\n--- Review queue ---")
    print(f"  Flagged            {ev['hybrid']['n_flagged']} of {ev['n_reviewed']} "
          f"({ev['flag_rate']:.1%} of expense entries)")
    print(f"  Value under review Rs "
          f"{scored.loc[scored['is_flagged'], 'amount'].sum():,.2f}")

    print(f"\n--- Measured against the {ev['n_planted']} planted errors ---")
    print(f"  {'Layer':<28}{'Flags':>7}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    for name, m in [("Rules only", ev["rules_only"]),
                    ("Isolation Forest only", ev["model_only"]),
                    ("Both (what ships)", ev["hybrid"])]:
        print(f"  {name:<28}{m['n_flagged']:>7}{m['precision']:>10.0%}"
              f"{m['recall']:>9.0%}{m['f1']:>8.3f}")
    print(f"  ROC-AUC {ev['roc_auc']:.3f}  (ranking quality of the model score, "
          f"independent of where the cut falls)")

    if ev["rules_only"]["f1"] > ev["hybrid"]["f1"]:
        print(f"\n  Note: rules alone score a slightly higher F1. Isolation Forest")
        print(f"  still ships, deliberately. Rules only catch error types somebody")
        print(f"  thought to write a rule for; the next error in a real ledger will")
        print(f"  be one nobody anticipated. The cost of keeping it is "
              f"{ev['hybrid']['n_flagged'] - ev['rules_only']['n_flagged']} extra rows")
        print(f"  in the queue.")

    print(f"\n--- Detection by error type ---")
    for _, row in ev["by_type"].iterrows():
        print(f"  {row['anomaly_type']:<16} {int(row['caught'])}/{int(row['planted'])} caught")

    if ev["missed"]:
        print(f"\n--- What it missed ({ev['missed']} of {ev['n_planted']}) ---")
        for _, m in ev["missed_rows"].head(4).iterrows():
            print(f"  {m['transaction_id']}  {m['category'][:22]:<22} "
                  f"Rs {m['amount']:>11,.2f}")
        print("  These sit inside the normal range for their ledger head. Catching")
        print("  them would mean flagging so much of the ledger that the queue")
        print("  stops being useful.")

    print(f"\n--- The {ev['false_positives']} false positives ---")
    fp = scored[scored["is_flagged"] & ~scored["transaction_id"].isin(ev["truth"])]
    for _, f in fp.head(4).iterrows():
        print(f"  {f['transaction_id']}  {f['category'][:22]:<22} "
              f"Rs {f['amount']:>11,.2f}  {f['reason'][:36]}")
    print("  Reviewing one costs an accountant perhaps thirty seconds. Missing a")
    print("  real error costs a misstated set of accounts. That asymmetry is why")
    print("  this system is tuned for recall rather than precision.")

    cols = ["transaction_id", "date", "description", "category", "vendor",
            "amount", "payment_method", "signals", "anomaly_score", "reason"]
    queue = scored[scored["is_flagged"]][cols].copy()
    queue["date"] = queue["date"].dt.date
    queue["anomaly_score"] = queue["anomaly_score"].round(4)
    queue["review_status"] = "Pending accountant review"
    queue.to_csv(OUTPUTS / "review_queue.csv", index=False)

    metrics = {
        "n_reviewed": ev["n_reviewed"], "n_planted": ev["n_planted"],
        "contamination": CONTAMINATION, "z_large": Z_LARGE, "z_weekend": Z_WEEKEND,
        "roc_auc": ev["roc_auc"], "flag_rate": ev["flag_rate"],
        "hybrid": ev["hybrid"], "rules_only": ev["rules_only"],
        "model_only": ev["model_only"],
        "true_positives": ev["true_positives"],
        "false_positives": ev["false_positives"], "missed": ev["missed"],
        "by_type": ev["by_type"].to_dict("records"),
    }
    (OUTPUTS / "anomaly_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved outputs/review_queue.csv ({len(queue)} rows for review)")
    print("\nCAVEAT: these figures measure detection of errors we planted ourselves.")
    print("They do NOT establish a fraud-detection accuracy. Real errors are not")
    print("drawn from the same distribution, and real fraud is designed to look")
    print("normal. Treat every flag as a question, not a conclusion.")


if __name__ == "__main__":
    main()
