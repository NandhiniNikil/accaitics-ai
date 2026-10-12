"""
Step 5 — "Ask Finance": plain-English questions, calculated answers.

THE ONE RULE
------------
A language model must NEVER produce a financial figure.

Ask an LLM "what were my March expenses?" and it will return a confident,
plausible, entirely invented number. In marketing that is annoying. In
accounting someone files a return on it.

So this module is built as:

    question
       -> INTENT MATCHING        (which kind of question is this?)
       -> PARAMETER EXTRACTION   (which month? which category?)
       -> CALCULATION in pandas  (the number comes from the ledger)
       -> TEMPLATED SENTENCE     (language only wraps a verified result)

Every answer carries a `workings` string showing the arithmetic, so the user
can check it. And when a question cannot be answered from the data, the
assistant says so rather than guessing — the refusal is a feature.

WHERE AN LLM WOULD FIT
----------------------
The intent layer below is keyword matching, which is transparent and runs
offline with no API key. FUNCTION_SCHEMA at the bottom exposes the same eight
calculations as JSON tool definitions. Drop those into an OpenAI or Anthropic
function-calling loop and the model replaces the keyword matcher while the
calculation layer stays exactly as it is. The trustworthy part does not change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from accaitics import accounting as acc

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})


@dataclass
class Answer:
    text: str
    workings: str = ""
    table: pd.DataFrame | None = None
    answered: bool = True


def _rs(n: float) -> str:
    return f"Rs {n:,.2f}"


def _compact(n: float) -> str:
    a = abs(n)
    s = "-" if n < 0 else ""
    if a >= 1e7:
        return f"{s}Rs {a/1e7:.2f} Cr"
    if a >= 1e5:
        return f"{s}Rs {a/1e5:.2f} L"
    return f"{s}Rs {a:,.0f}"


def _find_month(q: str, df: pd.DataFrame) -> pd.Period | None:
    periods = df["date"].dt.to_period("M")
    available = sorted(periods.unique())
    ql = q.lower()
    year = None
    ym = re.search(r"\b(20\d{2})\b", ql)
    if ym:
        year = int(ym.group(1))
    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", ql):
            cands = [p for p in available if p.month == num
                     and (year is None or p.year == year)]
            if cands:
                return cands[-1]
    if "last month" in ql or "latest" in ql:
        return available[-1]
    return None


def _find_category(q: str, df: pd.DataFrame) -> str | None:
    """Only EXPENSE heads — "spend on sales" must not match the income head."""
    ql = q.lower()
    expense_cats = df.loc[df["txn_type"] == "Expense", "category"].unique()
    for cat in expense_cats:
        if cat.lower() in ql:
            return cat
    # match on a distinctive word, e.g. "software" -> Software & Subscriptions
    for cat in expense_cats:
        for word in re.split(r"[\s&]+", cat.lower()):
            if len(word) > 4 and re.search(rf"\b{re.escape(word)}\w*", ql):
                return cat
    return None


# ---------------------------------------------------------------------------
# The calculations. Each returns an Answer built from real figures.
# ---------------------------------------------------------------------------

def q_revenue(df, month=None, **_) -> Answer:
    sub = df if month is None else df[df["date"].dt.to_period("M") == month]
    t = acc.totals(sub)
    period = f"in {month.strftime('%B %Y')}" if month else "over the whole period"
    return Answer(
        f"Total revenue {period} is {_compact(t['revenue'])}, from "
        f"{t['n_income']} income entries.",
        f"SUM(amount) WHERE txn_type = 'Income'\n= {_rs(t['revenue'])}")


def q_expenses(df, month=None, **_) -> Answer:
    sub = df if month is None else df[df["date"].dt.to_period("M") == month]
    t = acc.totals(sub)
    period = f"in {month.strftime('%B %Y')}" if month else "over the whole period"
    share = f" — {t['expenses']/t['revenue']*100:.1f}% of revenue" if t["revenue"] else ""
    return Answer(
        f"Total expenses {period} are {_compact(t['expenses'])} across "
        f"{t['n_expense']} entries{share}.",
        f"SUM(amount) WHERE txn_type = 'Expense'\n= {_rs(t['expenses'])}")


def q_profit(df, month=None, **_) -> Answer:
    sub = df if month is None else df[df["date"].dt.to_period("M") == month]
    t = acc.totals(sub)
    period = f"in {month.strftime('%B %Y')}" if month else "over the whole period"
    return Answer(
        f"Net profit {period} is {_compact(t['profit'])} on revenue of "
        f"{_compact(t['revenue'])} — a margin of {t['margin_pct']:.1f}%. "
        f"This is accounting profit, not your cash balance.",
        f"{_rs(t['revenue'])} revenue\n"
        f"- {_rs(t['expenses'])} expenses\n"
        f"= {_rs(t['profit'])} profit\n"
        f"margin = profit / revenue = {t['margin_pct']:.2f}%")


def q_spend_on(df, category=None, **_) -> Answer:
    if category is None:
        return Answer(
            "I could not tell which expense head you meant. Try naming it as it "
            "appears in your ledger, for example \"how much did I spend on "
            "software?\"", answered=False)
    sub = df[(df["category"] == category) & (df["txn_type"] == "Expense")]
    if sub.empty:
        return Answer(f"There are no expense entries under {category} in this "
                      f"ledger, so there is nothing to total.", answered=False)
    total = sub["amount"].sum()
    all_exp = acc.totals(df)["expenses"]
    if all_exp <= 0:
        return Answer(f"You spent {_compact(total)} on {category} across "
                      f"{len(sub)} entries. The ledger has no other expenses to "
                      f"compare that against.",
                      f"SUM(amount) WHERE category = '{category}'\n= {_rs(total)}")
    trend = acc.category_trend(df, category)
    return Answer(
        f"You spent {_compact(total)} on {category} across {len(sub)} entries — "
        f"{total/all_exp*100:.1f}% of total expenses. The largest single entry "
        f"was {_compact(sub['amount'].max())}.",
        f"SUM(amount) WHERE category = '{category}' AND txn_type = 'Expense'\n"
        f"= {_rs(total)} over {len(sub)} rows",
        table=trend)


def q_biggest_expense(df, **_) -> Answer:
    cats = acc.by_category(df, "Expense")
    if cats.empty:
        return Answer("There are no expense rows in this ledger.", answered=False)
    top = cats.iloc[0]
    lines = "\n".join(
        f"{i+1}. {r['category']} — {_compact(r['amount'])} ({r['share_pct']:.1f}%)"
        for i, r in cats.head(5).iterrows())
    return Answer(
        f"Your largest expense head is {top['category']} at "
        f"{_compact(top['amount'])}, which is {top['share_pct']:.1f}% of total "
        f"spend.\n\n{lines}",
        "GROUP BY category, SUM(amount) WHERE txn_type = 'Expense', ordered desc",
        table=cats)


def q_review(df, **_) -> Answer:
    from accaitics.anomaly import detect
    scored = detect(df)
    flagged = scored[scored["is_flagged"]]
    if flagged.empty:
        return Answer("Nothing stands out. No entry is far outside the normal "
                      "range for its own ledger head.")
    lines = "\n".join(
        f"- {r['date']:%d %b %Y} — {r['category']}, {_compact(r['amount'])}"
        f"  ({r['reason'].split(';')[0]})"
        for _, r in flagged.head(5).iterrows())
    return Answer(
        f"{len(flagged)} entries are worth a look, out of {len(scored)} expense "
        f"rows. The strongest signals first:\n\n{lines}\n\n"
        f"These are review items, not errors. Several will be legitimate.",
        "Rules (duplicate / round / weekend+large / category outlier) "
        "OR Isolation Forest, one-sided",
        table=flagged[["transaction_id", "date", "category", "amount", "reason"]])


def q_trend(df, **_) -> Answer:
    m = acc.monthly(df)
    if len(m) < 2:
        return Answer("There is only one month in this ledger, so no trend can "
                      "be measured.", answered=False)
    first, last = m.iloc[0], m.iloc[-1]
    best = m.loc[m["revenue"].idxmax()]
    if first["revenue"] <= 0:
        return Answer(
            f"The first month in this ledger ({first['month']}) has no recorded "
            f"revenue, so a percentage change from it cannot be computed. The "
            f"latest month ({last['month']}) shows {_compact(last['revenue'])}, "
            f"and the strongest month was {best['month']} at "
            f"{_compact(best['revenue'])}.",
            f"{first['month']} revenue = 0 — percentage change undefined",
            table=m)
    change = (last["revenue"] - first["revenue"]) / first["revenue"] * 100
    return Answer(
        f"Revenue moved from {_compact(first['revenue'])} in {first['month']} to "
        f"{_compact(last['revenue'])} in {last['month']} — "
        f"{'up' if change >= 0 else 'down'} {abs(change):.1f}%. The strongest "
        f"month was {best['month']} at {_compact(best['revenue'])}.",
        f"({_rs(last['revenue'])} - {_rs(first['revenue'])}) / "
        f"{_rs(first['revenue'])} x 100 = {change:.1f}%",
        table=m)


def q_vendors(df, **_) -> Answer:
    v = acc.top_vendors(df)
    if v.empty:
        return Answer("Your ledger has no vendor column filled in, so payments "
                      "cannot be grouped by payee.", answered=False)
    top = v.iloc[0]
    lines = "\n".join(
        f"{i+1}. {r['vendor']} — {_compact(r['amount'])} ({int(r['payments'])} payments)"
        for i, r in v.head(5).iterrows())
    return Answer(
        f"Your largest vendor is {top['vendor']} at {_compact(top['amount'])} "
        f"across {int(top['payments'])} payments.\n\n{lines}",
        "GROUP BY vendor, SUM(amount) WHERE txn_type = 'Expense'", table=v)


# ---------------------------------------------------------------------------
# Intent matching
# ---------------------------------------------------------------------------

INTENTS = [
    (lambda q: bool(re.search(r"\b(review|unusual|anomal|suspicious|error|duplicate|check)\b", q)),
     q_review, "review"),
    (lambda q: bool(re.search(r"spen[dt].*on|how much.*on\b|total.*on\b", q)),
     q_spend_on, "spend_on"),
    # Order-insensitive: "which expense category is highest" puts the
    # superlative AFTER the noun, and an earlier version missed it entirely.
    (lambda q: bool(re.search(r"\b(biggest|largest|highest|top|most)\b", q)
                    and re.search(r"\b(expense|cost|spend|spending|categor)", q)),
     q_biggest_expense, "biggest_expense"),
    (lambda q: bool(re.search(r"\b(vendors?|suppliers?|payees?)\b|paid most", q)),
     q_vendors, "vendors"),
    # Stems need no trailing \b or they never match inflected forms:
    # "chang" + \b cannot match "changed", which is how most people ask.
    (lambda q: bool(re.search(r"\b(trend|over time|month on month)\b|chang|grow", q)),
     q_trend, "trend"),
    (lambda q: bool(re.search(r"\b(profit|margin|bottom line|how.*doing)\b", q)),
     q_profit, "profit"),
    (lambda q: bool(re.search(r"\b(revenue|sales|turnover|income|earn)\b", q)),
     q_revenue, "revenue"),
    (lambda q: bool(re.search(r"\b(expense|spend|spent|cost|outgoing)\b", q)),
     q_expenses, "expenses"),
]

CANNOT_ANSWER = [
    (r"\b(receivable|owe|debtor|outstanding|unpaid)\b",
     "A transaction list cannot tell you what customers owe. Receivables need "
     "invoices with due dates and payment status, which this file does not have."),
    (r"\b(overdue|ageing|aging)\b",
     "Overdue analysis needs invoice due dates and payment dates. A transaction "
     "list does not contain them."),
    (r"\b(cash balance|bank balance|runway|how much cash)\b",
     "Accounting profit is not your cash position. Cash depends on the opening "
     "bank balance and which invoices have actually been paid — neither is in a "
     "transaction list."),
    (r"\b(tax|gst payable|tds|return)\b",
     "I do not compute tax liability. That depends on your registration, input "
     "credits and exemptions, none of which are in this file. Ask a qualified "
     "professional."),
    (r"\b(forecast|predict|next month|next year|will i)\b",
     "This assistant only reports what is in the ledger. It does not forecast."),
]

FALLBACK = (
    "I could not match that to a calculation I can run on your data. I can answer:\n\n"
    "- total revenue, expenses, profit and margin (optionally for one month)\n"
    "- how much was spent on a particular ledger head\n"
    "- which expense head is largest\n"
    "- which vendors were paid most\n"
    "- how revenue has moved over time\n"
    "- which entries are worth reviewing\n\n"
    "I do not guess. If I cannot compute an answer from your file, I say so "
    "rather than inventing a figure.")


def ask(question: str, df: pd.DataFrame) -> Answer:
    """Route a plain-English question to a calculation."""
    q = question.lower().strip()
    if not q:
        return Answer("Ask me something about your ledger.", answered=False)

    for pattern, message in CANNOT_ANSWER:
        if re.search(pattern, q):
            return Answer(message, answered=False)

    month = _find_month(q, df)
    category = _find_category(q, df)

    for test, fn, _name in INTENTS:
        if test(q):
            return fn(df, month=month, category=category)

    return Answer(FALLBACK, answered=False)


# ---------------------------------------------------------------------------
# The same capabilities as LLM function-calling definitions
# ---------------------------------------------------------------------------

FUNCTION_SCHEMA = [
    {"name": "total_revenue", "description": "Total income, optionally for one month.",
     "parameters": {"type": "object", "properties": {"month": {"type": "string", "description": "YYYY-MM"}}}},
    {"name": "total_expenses", "description": "Total expenses, optionally for one month.",
     "parameters": {"type": "object", "properties": {"month": {"type": "string", "description": "YYYY-MM"}}}},
    {"name": "profit_and_margin", "description": "Net profit and margin, optionally for one month.",
     "parameters": {"type": "object", "properties": {"month": {"type": "string", "description": "YYYY-MM"}}}},
    {"name": "spend_on_category", "description": "Total spent on one ledger head, with its monthly trend.",
     "parameters": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}},
    {"name": "biggest_expense_heads", "description": "Expense heads ranked by spend.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "transactions_to_review", "description": "Entries flagged as unusual for accountant review.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "revenue_trend", "description": "How revenue has moved month to month.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "top_vendors", "description": "Vendors ranked by total paid.",
     "parameters": {"type": "object", "properties": {}}},
]

SAMPLE_QUESTIONS = [
    "What is my total revenue?",
    "Which expense category is highest?",
    "How much did I spend on software?",
    "Which transactions should I review?",
    "What is my profit margin?",
    "How has revenue changed over time?",
    "Which vendors did I pay most?",
    "What was my profit in September?",
]


def main() -> None:
    from accaitics.accounting import load_ledger
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    df = load_ledger(root / "data" / "transactions.csv")

    print("=" * 70)
    print("STEP 5 — ASK FINANCE")
    print("=" * 70)
    print(f"\nLedger loaded: {len(df)} transactions. Every figure below is")
    print("calculated from that file. Nothing is generated.\n")

    demo = SAMPLE_QUESTIONS + ["How much do my customers owe me?",
                               "What is the capital of France?"]
    for q in demo:
        a = ask(q, df)
        print("-" * 70)
        print(f"Q: {q}")
        print(f"A: {a.text}")
        if a.workings:
            print(f"   [workings] {a.workings.splitlines()[0]}")
        print()

    print("-" * 70)
    print("Note the last two. Refusing to answer is a feature: an accounting")
    print("assistant that improvises is worse than no assistant at all.")


if __name__ == "__main__":
    main()
