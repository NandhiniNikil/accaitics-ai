"""
Tests for the accounting core.

These matter more than the ML tests. If the arithmetic is wrong, a confident
model on top of it is worse than useless.

The reconciliation tests re-compute every headline figure a SECOND time, by a
different route (raw Python sums over the CSV, and SQL over SQLite), and assert
all three agree to the paisa. That is the programmatic version of tying your
report back to the trial balance.
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accaitics import accounting as acc  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ledger():
    return acc.load_ledger(DATA / "transactions.csv")


# ---------------------------------------------------------------------------
# Reconciliation — the same number computed three independent ways
# ---------------------------------------------------------------------------

def test_revenue_reconciles_with_raw_python(ledger):
    """pandas total must equal a hand-rolled sum over the raw CSV."""
    manual = 0.0
    with open(DATA / "transactions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["txn_type"] == "Income":
                manual += float(row["amount"])
    assert acc.totals(ledger)["revenue"] == pytest.approx(manual, abs=0.01)


def test_expenses_reconcile_with_raw_python(ledger):
    manual = 0.0
    with open(DATA / "transactions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["txn_type"] == "Expense":
                manual += float(row["amount"])
    assert acc.totals(ledger)["expenses"] == pytest.approx(manual, abs=0.01)


def test_totals_reconcile_with_sql(ledger):
    """The SQLite copy must agree with the pandas calculation."""
    con = sqlite3.connect(DATA / "accaitics.db")
    rev = con.execute(
        "SELECT SUM(amount) FROM transactions WHERE txn_type='Income'").fetchone()[0]
    exp = con.execute(
        "SELECT SUM(amount) FROM transactions WHERE txn_type='Expense'").fetchone()[0]
    con.close()
    t = acc.totals(ledger)
    assert t["revenue"] == pytest.approx(rev, abs=0.01)
    assert t["expenses"] == pytest.approx(exp, abs=0.01)


def test_profit_reconciles_with_raw_python(ledger):
    """Profit computed independently from the CSV, not from totals() itself."""
    inc = exp = 0.0
    with open(DATA / "transactions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["txn_type"] == "Income":
                inc += float(row["amount"])
            else:
                exp += float(row["amount"])
    assert acc.totals(ledger)["profit"] == pytest.approx(inc - exp, abs=0.01)


def test_margin_reconciles_with_raw_python(ledger):
    inc = exp = 0.0
    with open(DATA / "transactions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["txn_type"] == "Income":
                inc += float(row["amount"])
            else:
                exp += float(row["amount"])
    assert acc.totals(ledger)["margin_pct"] == pytest.approx((inc - exp) / inc * 100,
                                                             abs=1e-6)


def test_aggregations_agree_when_a_sign_error_is_present():
    """
    REGRESSION TEST. totals() used .abs() while monthly() and by_category()
    summed the raw signed amount, so any ledger containing a sign error made
    the KPI tile disagree with the chart directly beneath it, silently.
    The synthetic ledger has no negatives, so only an injected one catches it.
    """
    df = acc.load_ledger(DATA / "transactions.csv").copy()
    idx = df.index[df["txn_type"] == "Expense"][5]
    df.loc[idx, "amount"] = -abs(df.loc[idx, "amount"])

    t = acc.totals(df)
    assert acc.monthly(df)["expenses"].sum() == pytest.approx(t["expenses"], abs=0.01)
    assert acc.by_category(df, "Expense")["amount"].sum() == pytest.approx(
        t["expenses"], abs=0.01)


def test_category_breakdown_sums_to_total_expenses(ledger):
    """Every rupee of expense must land in exactly one ledger head."""
    cats = acc.by_category(ledger, "Expense")
    assert cats["amount"].sum() == pytest.approx(acc.totals(ledger)["expenses"], abs=0.01)
    # Shares checked against independently grouped sums, not against their
    # own definition.
    manual = (ledger[ledger["txn_type"] == "Expense"]
              .groupby("category")["amount"].sum())
    total = manual.sum()
    for _, row in cats.iterrows():
        assert row["share_pct"] == pytest.approx(
            manual[row["category"]] / total * 100, abs=1e-6)


def test_monthly_sums_to_annual(ledger):
    """Twelve monthly figures must add back to the yearly total."""
    m = acc.monthly(ledger)
    t = acc.totals(ledger)
    assert m["revenue"].sum() == pytest.approx(t["revenue"], abs=0.01)
    assert m["expenses"].sum() == pytest.approx(t["expenses"], abs=0.01)
    assert m["profit"].sum() == pytest.approx(t["profit"], abs=0.01)


def test_entry_counts_add_up(ledger):
    t = acc.totals(ledger)
    assert t["n_income"] + t["n_expense"] == t["n_total"] == len(ledger)


# ---------------------------------------------------------------------------
# Worked example — arithmetic anyone can check on paper
# ---------------------------------------------------------------------------

def _tiny():
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "date": pd.to_datetime(["2025-04-05", "2025-04-18", "2025-05-02", "2025-05-20"]),
        "description": ["Sales invoice", "Rent paid", "Sales invoice", "Fuel"],
        "amount": [100000.0, 25000.0, 150000.0, 5000.0],
        "txn_type": ["Income", "Expense", "Income", "Expense"],
        "category": ["Sales Revenue", "Rent", "Sales Revenue", "Travel & Fuel"],
        "payment_method": ["Bank Transfer"] * 4,
        "vendor": ["", "Landlord", "", "Indian Oil"],
    })


def test_monthly_emits_no_warnings_on_zero_revenue():
    """
    REGRESSION TEST for a portability bug.

    monthly() used `.replace(0, pd.NA) ... .fillna(0.0)` to guard the margin
    division. That pushes the column to object dtype and back, which pandas 2.x
    warns about on the downcast. It raised nothing on the version this was
    developed against and only appeared when it was run on another machine.
    """
    import warnings
    df = _tiny().copy()
    df.loc[df["txn_type"] == "Income", "txn_type"] = "Expense"   # zero revenue
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = acc.monthly(df)
    assert (m["margin_pct"] == 0.0).all()
    assert not m["margin_pct"].isna().any()


def test_margin_is_zero_not_nan_when_a_month_has_no_revenue():
    df = _tiny().copy()
    df.loc[0, "txn_type"] = "Expense"        # April now has no income
    m = acc.monthly(df).set_index("month")
    assert m.loc["2025-04", "margin_pct"] == 0.0
    assert m.loc["2025-05", "margin_pct"] > 0


def test_worked_example_totals():
    """Revenue 250,000 − expenses 30,000 = profit 220,000, margin 88%."""
    t = acc.totals(_tiny())
    assert t["revenue"] == 250000.0
    assert t["expenses"] == 30000.0
    assert t["profit"] == 220000.0
    assert t["margin_pct"] == pytest.approx(88.0)


def test_worked_example_monthly():
    m = acc.monthly(_tiny()).set_index("month")
    assert m.loc["2025-04", "profit"] == 75000.0    # 100,000 − 25,000
    assert m.loc["2025-05", "profit"] == 145000.0   # 150,000 −  5,000


# ---------------------------------------------------------------------------
# Validation — bad input must fail loudly, not silently
# ---------------------------------------------------------------------------

def test_missing_required_column_is_rejected():
    df = _tiny().drop(columns=["amount"])
    _, res = acc.validate(df)
    assert res.ok is False
    assert any("amount" in e for e in res.errors)


def test_unreadable_date_row_is_dropped_and_reported():
    """A real upload arrives as text, so the bad value is a string here too."""
    df = _tiny()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.loc[0, "date"] = "not a date"
    clean, res = acc.validate(df)
    assert len(clean) == 3
    assert any("unreadable date" in w for w in res.warnings)


def test_iso_dates_are_not_mangled_by_dayfirst():
    """
    REGRESSION TEST for a real bug.

    An earlier version parsed with dayfirst=True and format="mixed", which
    turned 2025-04-01 into 4 January 2025. 305 of 788 rows were silently
    corrupted and every monthly report was wrong, with no error raised.
    """
    s = pd.Series(["2025-04-01", "2025-12-31", "2026-03-15"])
    out = acc.parse_dates(s)
    assert out.iloc[0] == pd.Timestamp("2025-04-01")   # April, not January
    assert out.iloc[0].month == 4
    assert out.iloc[1] == pd.Timestamp("2025-12-31")
    assert out.iloc[2] == pd.Timestamp("2026-03-15")


def test_full_ledger_stays_inside_its_financial_year(ledger):
    """The generated ledger covers Apr 2025 - Mar 2026 and nothing outside."""
    months = sorted(ledger["date"].dt.to_period("M").astype(str).unique())
    assert months[0] == "2025-04"
    assert months[-1] == "2026-03"
    assert len(months) == 12


def test_day_first_dates_still_work():
    """DD/MM/YYYY is the Indian convention and must still parse."""
    out = acc.parse_dates(pd.Series(["15/03/2026", "01/04/2025"]))
    assert out.iloc[0] == pd.Timestamp("2026-03-15")
    assert out.iloc[1] == pd.Timestamp("2025-04-01")


def test_mixed_date_formats_are_parsed():
    """Indian exports commonly use DD/MM/YYYY; ISO also has to work."""
    df = _tiny()
    df["date"] = ["05/04/2025", "2025-04-18", "02/05/2025", "2025-05-20"]
    clean, res = acc.validate(df)
    assert res.ok is True
    assert len(clean) == 4
    assert clean["date"].min() == pd.Timestamp("2025-04-05")


def test_currency_symbols_and_commas_are_parsed():
    df = _tiny()
    df["amount"] = df["amount"].astype(str)
    df.loc[0, "amount"] = "₹1,00,000.00"
    clean, res = acc.validate(df)
    assert res.ok
    assert clean.loc[clean["transaction_id"] == "T1", "amount"].iloc[0] == 100000.0


def test_duplicate_transaction_ids_are_flagged():
    df = _tiny()
    df.loc[3, "transaction_id"] = "T1"
    _, res = acc.validate(df)
    assert "T1" in res.duplicate_ids
    assert any("more than once" in w for w in res.warnings)


def test_column_aliases_are_recognised():
    """A ledger exported with different headers should still work."""
    df = _tiny().rename(columns={
        "transaction_id": "Voucher No", "date": "Posting Date",
        "description": "Narration", "amount": "Amt", "txn_type": "Type"})
    clean, res = acc.validate(df)
    assert res.ok is True
    assert acc.totals(clean)["revenue"] == 250000.0


def test_type_variants_are_normalised():
    df = _tiny()
    df["txn_type"] = ["Credit", "Debit", "sale", "payment"]
    clean, res = acc.validate(df)
    assert set(clean["txn_type"]) == {"Income", "Expense"}
    assert acc.totals(clean)["revenue"] == 250000.0


def test_empty_file_fails_cleanly():
    df = pd.DataFrame(columns=["transaction_id", "date", "description",
                               "amount", "txn_type"])
    _, res = acc.validate(df)
    assert res.ok is False
