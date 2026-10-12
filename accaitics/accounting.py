"""
Step 2 — Validation and verified financial calculations.

This module is the accounting core. It is deliberately the FIRST thing built
and the most heavily tested, because the whole project rests on it: if the
totals are wrong, nothing downstream matters.

Every function here is plain arithmetic with no machine learning. That is the
point — financial figures must come from deterministic calculation, never from
a model's opinion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REQUIRED = ["transaction_id", "date", "description", "amount", "txn_type"]
OPTIONAL = ["category", "payment_method", "vendor"]

# Column names a real export might use instead of ours.
ALIASES = {
    "transaction_id": ["txn_id", "id", "voucher_no", "reference", "ref", "entry_no"],
    "date": ["txn_date", "transaction_date", "posting_date", "entry_date", "dt"],
    "description": ["narration", "particulars", "details", "memo", "remarks"],
    "amount": ["amt", "value", "transaction_amount", "net_amount", "total"],
    "txn_type": ["type", "transaction_type", "dr_cr", "direction", "entry_type"],
    "category": ["account", "ledger", "head", "account_head", "expense_category"],
    "payment_method": ["mode", "payment_mode", "method"],
    "vendor": ["party", "supplier", "payee", "vendor_name"],
}


@dataclass
class ValidationResult:
    """What validation found. `ok` False means do not proceed."""
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_in: int = 0
    rows_usable: int = 0
    duplicate_ids: list[str] = field(default_factory=list)
    renamed: dict[str, str] = field(default_factory=dict)


def _normalise(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename recognised variants to our canonical names."""
    df = df.copy()
    current = {_normalise(c): c for c in df.columns}
    renamed = {}
    for canon, alts in ALIASES.items():
        if canon in current:
            continue
        for alt in alts:
            if alt in current:
                renamed[current[alt]] = canon
                break
    if renamed:
        df = df.rename(columns=renamed)
    df.columns = [_normalise(c) for c in df.columns]
    return df, renamed


def parse_dates(series: pd.Series) -> pd.Series:
    """
    Parse a date column safely.

    THIS FUNCTION EXISTS BECAUSE OF A REAL BUG. The first version used
    `pd.to_datetime(..., format="mixed", dayfirst=True)`, reasoning that Indian
    exports usually write DD/MM/YYYY. That silently reinterpreted ISO dates:
    2025-04-01 became 4 January 2025. On a 788-row ledger it corrupted 305
    rows, and because every date still looked valid, nothing raised an error —
    the monthly reports were simply wrong.

    The fix is to resolve the unambiguous format first:
      1. ISO (YYYY-MM-DD) — never ambiguous, so try it before anything else.
      2. Whatever is left is day-first (DD/MM/YYYY), the Indian convention.
      3. Anything still unparsed becomes NaT and is dropped with a warning.
    """
    s = series.astype(str).str.strip()

    # Already datetimes? Leave them alone.
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    out = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")

    remaining = out.isna() & s.ne("") & s.ne("nan")
    if remaining.any():
        fallback = pd.to_datetime(s[remaining], errors="coerce",
                                  dayfirst=True, format="mixed")
        out.loc[remaining] = fallback

    return out


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
    """
    Check an uploaded ledger and return (clean_df, result).

    Rows that cannot be used are dropped and counted rather than silently
    coerced — an accountant needs to know what was thrown away.
    """
    res = ValidationResult(ok=True, rows_in=len(df))
    df, renamed = map_columns(df)
    res.renamed = renamed

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        res.ok = False
        res.errors.append(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(df.columns)}")
        return df, res

    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = ""
            res.warnings.append(f"No '{c}' column — filled with blanks.")

    # --- dates ---
    df["date"] = parse_dates(df["date"])
    bad_dates = int(df["date"].isna().sum())
    if bad_dates:
        res.warnings.append(f"{bad_dates} row(s) dropped — unreadable date.")
        df = df[df["date"].notna()]

    # --- amounts ---
    df["amount"] = (df["amount"].astype(str)
                    .str.replace(r"[₹$£€,\s]", "", regex=True)
                    .str.replace(r"^\((.*)\)$", r"-\1", regex=True))
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    bad_amt = int(df["amount"].isna().sum())
    if bad_amt:
        res.warnings.append(f"{bad_amt} row(s) dropped — unreadable amount.")
        df = df[df["amount"].notna()]

    zero = int((df["amount"] == 0).sum())
    if zero:
        res.warnings.append(f"{zero} row(s) have a zero amount — kept, but check them.")

    neg = int((df["amount"] < 0).sum())
    if neg:
        res.warnings.append(
            f"{neg} row(s) have a negative amount. In double-entry books direction "
            f"belongs in the type column, so these are treated as sign errors and "
            f"flagged for review.")

    # --- transaction type ---
    df["txn_type"] = df["txn_type"].astype(str).str.strip().str.title()
    df.loc[df["txn_type"].str.contains("inc|rev|sale|credit|cr$|receipt",
                                       case=False, regex=True), "txn_type"] = "Income"
    df.loc[df["txn_type"].str.contains("exp|debit|dr$|payment|purchase",
                                       case=False, regex=True), "txn_type"] = "Expense"
    unknown = ~df["txn_type"].isin(["Income", "Expense"])
    if unknown.any():
        res.warnings.append(
            f"{int(unknown.sum())} row(s) dropped — type was neither Income nor Expense.")
        df = df[~unknown]

    # --- duplicate transaction IDs ---
    dupes = df[df.duplicated("transaction_id", keep=False)]
    if len(dupes):
        ids = sorted(dupes["transaction_id"].unique().tolist())
        res.duplicate_ids = ids
        res.warnings.append(
            f"{len(ids)} transaction ID(s) appear more than once. These are kept "
            f"but flagged — a repeated voucher number usually means the same bill "
            f"was entered twice.")

    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df.loc[df["category"] == "", "category"] = "Uncategorised"
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    df = df.sort_values("date").reset_index(drop=True)
    res.rows_usable = len(df)
    if res.rows_usable == 0:
        res.ok = False
        res.errors.append("No usable rows left after validation.")
    return df, res


# ---------------------------------------------------------------------------
# Financial calculations — the figures a business owner actually asks for
# ---------------------------------------------------------------------------

def totals(df: pd.DataFrame) -> dict:
    """
    Headline P&L figures.

    Amounts are taken as absolute values because direction is carried by
    txn_type. A negative amount is a data-entry error, not a credit.
    """
    inc = df.loc[df["txn_type"] == "Income", "amount"].abs().sum()
    exp = df.loc[df["txn_type"] == "Expense", "amount"].abs().sum()
    profit = inc - exp
    return {
        "revenue": float(inc),
        "expenses": float(exp),
        "profit": float(profit),
        "margin_pct": float(profit / inc * 100) if inc else 0.0,
        "n_income": int((df["txn_type"] == "Income").sum()),
        "n_expense": int((df["txn_type"] == "Expense").sum()),
        "n_total": int(len(df)),
        "from": df["date"].min(),
        "to": df["date"].max(),
    }


def by_category(df: pd.DataFrame, txn_type: str = "Expense") -> pd.DataFrame:
    """
    Spend per ledger head, largest first, with share of total.

    Uses .abs() for the same reason totals() does: direction is carried by
    txn_type, so a negative amount is a sign error rather than a credit. Every
    aggregation in this module must agree on that, or the KPI tile and the
    chart beneath it will disagree on any ledger containing a sign error.
    """
    sub = df[df["txn_type"] == txn_type].copy()
    sub["amount"] = sub["amount"].abs()
    if sub.empty:
        return pd.DataFrame(columns=["category", "amount", "share_pct", "entries"])
    g = (sub.groupby("category")["amount"]
         .agg(amount="sum", entries="count")
         .reset_index()
         .sort_values("amount", ascending=False))
    g["share_pct"] = g["amount"] / g["amount"].sum() * 100
    return g[["category", "amount", "share_pct", "entries"]].reset_index(drop=True)


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Month-by-month income, expense, profit and margin."""
    d = df.copy()
    d["amount"] = d["amount"].abs()        # see by_category() for why
    d["month"] = d["date"].dt.to_period("M")
    g = (d.pivot_table(index="month", columns="txn_type", values="amount",
                       aggfunc="sum", fill_value=0.0)
         .rename(columns={"Income": "revenue", "Expense": "expenses"}))
    for c in ("revenue", "expenses"):
        if c not in g.columns:
            g[c] = 0.0
    g["profit"] = g["revenue"] - g["expenses"]
    g["margin_pct"] = (g["profit"] / g["revenue"].replace(0, pd.NA) * 100).fillna(0.0)
    g = g.reset_index()
    g["month"] = g["month"].astype(str)
    return g[["month", "revenue", "expenses", "profit", "margin_pct"]]


def category_trend(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """One ledger head tracked over the months."""
    sub = df[df["category"] == category].copy()
    if sub.empty:
        return pd.DataFrame(columns=["month", "amount"])
    sub["amount"] = sub["amount"].abs()     # see by_category() for why
    sub["month"] = sub["date"].dt.to_period("M").astype(str)
    return (sub.groupby("month")["amount"].sum().reset_index()
            .sort_values("month").reset_index(drop=True))


def top_vendors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    sub = df[(df["txn_type"] == "Expense") &
             (df["vendor"].astype(str).str.len() > 0)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["vendor", "amount", "payments"])
    sub["amount"] = sub["amount"].abs()     # see by_category() for why
    return (sub.groupby("vendor")["amount"]
            .agg(amount="sum", payments="count")
            .reset_index()
            .sort_values("amount", ascending=False)
            .head(n).reset_index(drop=True))


def load_ledger(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel ledger and validate it. Raises on fatal problems."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)
    df, res = validate(raw)
    if not res.ok:
        raise ValueError("; ".join(res.errors))
    return df
