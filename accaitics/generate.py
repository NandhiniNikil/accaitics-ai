"""
Step 1 — Build the accounting data layer.

Creates a synthetic transaction ledger for a fictional small business,
"Vaigai Traders Pvt Ltd" (Erode, Tamil Nadu).

WHY SYNTHETIC DATA
------------------
Two reasons, and the second matters more:
  1. There is no real client ledger to practise on.
  2. Real client books must NEVER be pushed to GitHub or fed to public AI
     tools. Synthetic data lets the method be published without the data.

WHAT EACH COLUMN MEANS (the accounting logic)
---------------------------------------------
  transaction_id  Unique reference, the way Tally gives every voucher a number.
                  Used to detect duplicate entry — the same bill keyed twice.
  date            Posting date. Drives every period calculation.
  description     What the bookkeeper actually typed. Deliberately messy:
                  abbreviations, ALL CAPS, truncation. This is the input to
                  the categorisation model, so clean text would make the
                  problem artificially easy.
  amount          Always POSITIVE. Direction is carried by txn_type, which is
                  how a double-entry system works — you do not record a
                  negative debit, you record a credit.
  txn_type        Income or Expense.
  category        The ledger account the entry belongs to. This is the LABEL
                  the categorisation model learns to predict.
  payment_method  Bank Transfer / UPI / Cheque / Credit Card / Cash.
  vendor          Who was paid. Blank for income rows.

PLANTED ERRORS
--------------
About 3% of all rows are deliberately corrupted and recorded in
data/anomaly_key.csv. The ML model NEVER sees that file — it exists only so
detection can be scored honestly afterwards. A real ledger has no such key,
which is exactly why anomaly detection has to be unsupervised.

Run:  python -m accaitics.generate
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261012)   # fixed seed -> identical output every run

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2025-04-01")      # Indian financial year
END = pd.Timestamp("2026-03-31")

# ---------------------------------------------------------------------------
# The chart of accounts. share = how much of monthly revenue this head eats.
# ---------------------------------------------------------------------------
EXPENSE_HEADS = [
    # (category, share_of_revenue, fixed_amount, vendors, description templates)
    ("Cost of Goods Sold", 0.44, None,
     ["Sri Balaji Suppliers", "Reliance Wholesale", "Coimbatore Traders"],
     ["Purchase of raw material inv {r}", "RM purchase bill no {r}",
      "Stock purchase - {r}", "Goods inward GRN {r}", "Raw material bill {r}"]),
    ("Salaries & Wages", None, 385000,
     [None],
     ["Salary payment for the month", "Monthly payroll disbursement",
      "Staff salary transfer payroll {r}"]),
    ("Rent", None, 68000,
     ["Perundurai Rentals"],
     ["Office rent for the month", "Godown rent payment {r}", "Monthly rent premises"]),
    ("Utilities", 0.011, None,
     ["TNEB", "Airtel Business"],
     ["Electricity bill payment {r}", "EB bill service no {r}",
      "Internet and phone bill {r}", "Power charges {r}"]),
    ("Marketing & Advertising", 0.038, None,
     ["Google Ads", "Meta Platforms"],
     ["Google Ads spend {r}", "Facebook ads billing {r}",
      "PPC ad spend - {r}", "Instagram campaign spend"]),
    ("Shipping & Freight", 0.027, None,
     ["BlueDart Express", "Delhivery", "Chennai Freight Lines"],
     ["Courier charges {r}", "Freight outward bill {r}",
      "Delivery charges AWB {r}", "Transport charges LR {r}"]),
    ("Software & Subscriptions", None, 24500,
     ["Zoho Corp", "Microsoft India", "Tally Solutions"],
     ["Zoho subscription renewal", "Microsoft 365 subscription {r}",
      "Tally AMC renewal {r}", "CRM software monthly fee {r}"]),
    ("Office Supplies", 0.006, None,
     ["Amazon Business", "Kumar Stationery"],
     ["Office supplies purchase {r}", "Stationery and consumables {r}",
      "Printing and stationery {r}", "Amazon business order {r}"]),
    ("Travel & Fuel", 0.009, None,
     ["Indian Oil", None],
     ["Fuel expenses for the month {r}", "Diesel purchase vehicle {r}",
      "Petrol reimbursement {r}", "Travel expenses {r}"]),
    ("Insurance", None, 15000,
     ["HDFC Ergo"],
     ["Insurance premium payment {r}", "Fire and burglary policy premium"]),
    ("Bank Charges & Interest", 0.007, None,
     ["ICICI Bank"],
     ["Bank charges and interest {r}", "Loan EMI interest portion {r}",
      "Processing fee {r}"]),
    ("Repairs & Maintenance", 0.006, None,
     ["Ravi Electricals", None],
     ["Machine repair charges {r}", "AMC and maintenance {r}",
      "Electrical repair work {r}"]),
    ("Professional Fees", 0.005, None,
     [None],
     ["Audit fees payment {r}", "CA consultation charges {r}",
      "Legal and professional fee {r}"]),
]

CUSTOMERS = ["Anand Enterprises", "Bharat Textiles", "Chola Hardware",
             "Deccan Logistics", "Everest Foods", "Ganga Motors",
             "Harvest Agro", "Indus Steel", "Kaveri Mills", "Lotus Interiors"]

# Bookkeepers type like this. Clean text would make categorisation trivial.
ABBREV = {"payment": "pymt", "purchase": "pur", "invoice": "inv",
          "charges": "chrgs", "monthly": "mthly", "expenses": "exp",
          "maintenance": "maint", "and": "&", "bill": "bl",
          "subscription": "subscr", "material": "matl", "stationery": "stny"}

# Entries so vague that a human would need the vendor or amount to code them.
# These set a realistic ceiling on what any text model can achieve.
AMBIGUOUS = ["misc expenses", "payment made", "bill paid", "sundry expenses",
             "paid to party {r}", "general expenses", "other charges",
             "adjustment entry {r}", "cash payment {r}"]

SEASON = {1: .95, 2: .90, 3: 1.05, 4: .92, 5: 1.0, 6: 1.02,
          7: 1.05, 8: 1.08, 9: 1.22, 10: 1.30, 11: 1.15, 12: 1.04}


def _ref() -> str:
    return str(RNG.integers(1000, 99999))


def messify(text: str) -> str:
    """Apply realistic bookkeeping sloppiness."""
    r = RNG.random()
    if r < 0.20:
        for full, short in ABBREV.items():
            if RNG.random() < 0.5:
                text = text.replace(full, short)
    if r < 0.10:
        text = text.upper()
    elif 0.10 <= r < 0.16:
        text = text.lower()
    if RNG.random() < 0.07:
        text = text[:max(12, int(len(text) * 0.65))].strip()
    return text


def build_ledger() -> pd.DataFrame:
    rows = []
    tid = 10000
    months = pd.date_range(START, END, freq="MS")

    for k, month in enumerate(months):
        trend = (1 + 0.009) ** k
        revenue = 1_850_000 * trend * SEASON[month.month] * RNG.normal(1.0, 0.05)

        # ---- income rows ----
        n_sales = int(RNG.integers(22, 32))
        shares = RNG.dirichlet(np.ones(n_sales))
        for s in shares:
            tid += 1
            day = int(RNG.integers(1, month.days_in_month + 1))
            cust = str(RNG.choice(CUSTOMERS))
            rows.append({
                "transaction_id": f"TXN{tid}",
                "date": month + pd.Timedelta(days=day - 1),
                "description": f"Sales invoice INV{2000 + tid % 9000} - {cust}",
                "amount": round(float(revenue * s), 2),
                "txn_type": "Income",
                "category": "Sales Revenue",
                "payment_method": str(RNG.choice(["Bank Transfer", "UPI", "Cheque"],
                                                 p=[.6, .25, .15])),
                "vendor": "",
            })

        # ---- expense rows ----
        for cat, share, fixed, vendors, templates in EXPENSE_HEADS:
            if fixed is not None:
                total = fixed * RNG.normal(1.0, 0.015)
                n_lines = 1
            else:
                total = revenue * share * RNG.normal(1.0, 0.11)
                n_lines = int(RNG.integers(2, 7))

            parts = RNG.dirichlet(np.ones(n_lines))
            for p in parts:
                amt = round(float(total * p), 2)
                if amt <= 0:
                    continue
                tid += 1
                day = int(RNG.integers(1, month.days_in_month + 1))
                # 6% of entries are genuinely ambiguous
                if RNG.random() < 0.06:
                    desc = str(RNG.choice(AMBIGUOUS)).replace("{r}", _ref())
                else:
                    desc = messify(str(RNG.choice(templates)).replace("{r}", _ref()))
                v = RNG.choice([x for x in vendors])
                rows.append({
                    "transaction_id": f"TXN{tid}",
                    "date": month + pd.Timedelta(days=day - 1),
                    "description": desc,
                    "amount": amt,
                    "txn_type": "Expense",
                    "category": cat,
                    "payment_method": str(RNG.choice(
                        ["Bank Transfer", "UPI", "Cheque", "Credit Card", "Cash"],
                        p=[.42, .26, .12, .14, .06])),
                    "vendor": "" if v is None else str(v),
                })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def inject_anomalies(df: pd.DataFrame, rate: float = 0.03) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Plant realistic bookkeeping errors and return (ledger, answer_key).

    The five error types are the ones that actually happen in practice:
      duplicate      — the same bill keyed twice
      fat_finger     — an extra zero typed (4,500 entered as 45,000)
      round_number   — a suspiciously round manual entry
      weekend_large  — a large payment posted on a Sunday
      outlier_amount — far above the normal range for that head

    Keying errors cluster on small-to-mid value lines in real life, so
    candidates are drawn from below the 70th percentile. Letting a 10x
    fat-finger land on the largest COGS bill would distort the whole P&L
    and make detection trivially easy.
    """
    df = df.copy()
    expenses = df[df["txn_type"] == "Expense"]
    cutoff = expenses["amount"].quantile(0.70)
    pool = expenses.index[expenses["amount"] <= cutoff].to_numpy()

    n_bad = int(len(df) * rate)
    chosen = RNG.choice(pool, size=min(n_bad, len(pool)), replace=False)
    kinds = RNG.choice(
        ["duplicate", "fat_finger", "round_number", "weekend_large", "outlier_amount"],
        size=len(chosen), p=[.26, .22, .14, .16, .22])

    key_rows, new_rows = [], []
    next_tid = 90000

    for idx, kind in zip(chosen, kinds):
        row = df.loc[idx]
        if kind == "duplicate":
            dup = row.copy()
            next_tid += 1
            dup["transaction_id"] = f"TXN{next_tid}"
            dup["date"] = row["date"] + pd.Timedelta(days=int(RNG.integers(0, 3)))
            new_rows.append(dup)
            # BOTH halves of the pair are the finding. The rule flags both (it
            # cannot know which is the original), so recording only the
            # injected copy would score every correct catch as half a false
            # positive and understate precision by ~20 points.
            key_rows.append({"transaction_id": dup["transaction_id"],
                             "anomaly_type": "duplicate"})
            key_rows.append({"transaction_id": row["transaction_id"],
                             "anomaly_type": "duplicate"})
        elif kind == "fat_finger":
            df.loc[idx, "amount"] = round(float(row["amount"]) * 10, 2)
            key_rows.append({"transaction_id": row["transaction_id"],
                             "anomaly_type": "fat_finger"})
        elif kind == "round_number":
            df.loc[idx, "amount"] = float(RNG.choice([25000, 50000, 75000, 100000]))
            key_rows.append({"transaction_id": row["transaction_id"],
                             "anomaly_type": "round_number"})
        elif kind == "weekend_large":
            d = pd.Timestamp(row["date"])
            d = d + pd.Timedelta(days=(6 - d.weekday()) % 7)
            df.loc[idx, "date"] = d
            df.loc[idx, "amount"] = round(float(row["amount"]) * float(RNG.uniform(2.5, 4.5)), 2)
            key_rows.append({"transaction_id": row["transaction_id"],
                             "anomaly_type": "weekend_large"})
        else:
            df.loc[idx, "amount"] = round(float(row["amount"]) * float(RNG.uniform(5, 9)), 2)
            key_rows.append({"transaction_id": row["transaction_id"],
                             "anomaly_type": "outlier_amount"})

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df = df.sort_values("date").reset_index(drop=True)
    return df, pd.DataFrame(key_rows)


def write_sqlite(df: pd.DataFrame) -> Path:
    """Store the ledger in SQLite so the SQL half of the stack is real."""
    path = DATA / "accaitics.db"
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_sql("transactions", con, index=False)
    con.execute("CREATE INDEX idx_date ON transactions(date)")
    con.execute("CREATE INDEX idx_cat ON transactions(category)")
    con.commit()
    con.close()
    return path


def main() -> None:
    print("=" * 70)
    print("STEP 1 — BUILDING THE ACCOUNTING DATA LAYER")
    print("=" * 70)

    ledger = build_ledger()
    print(f"\nGenerated {len(ledger)} clean transactions "
          f"({START:%d %b %Y} to {END:%d %b %Y})")

    ledger, key = inject_anomalies(ledger)
    print(f"Planted {len(key)} deliberate errors "
          f"({len(key)/len(ledger)*100:.1f}% of {len(ledger)} rows)")
    for t, n in key["anomaly_type"].value_counts().items():
        print(f"    {t:<16} {n}")

    ledger.to_csv(DATA / "transactions.csv", index=False)
    key.to_csv(DATA / "anomaly_key.csv", index=False)
    db = write_sqlite(ledger)

    inc = ledger.loc[ledger["txn_type"] == "Income", "amount"].sum()
    exp = ledger.loc[ledger["txn_type"] == "Expense", "amount"].sum()
    print(f"\nTotal revenue   Rs {inc:>14,.2f}")
    print(f"Total expenses  Rs {exp:>14,.2f}")
    print(f"Net profit      Rs {inc - exp:>14,.2f}   ({(inc-exp)/inc*100:.1f}% margin)")
    print(f"\nWritten:")
    print(f"  data/transactions.csv   {len(ledger)} rows")
    print(f"  data/anomaly_key.csv    {len(key)} rows  (answer key — models never see this)")
    print(f"  data/accaitics.db       SQLite, table 'transactions'")


if __name__ == "__main__":
    main()
