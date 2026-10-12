-- ===========================================================================
-- ACCAITICS AI — SQL layer
-- ===========================================================================
-- Run:  sqlite3 data/accaitics.db < sql/queries.sql
--       (or open data/accaitics.db in DB Browser for SQLite)
--
-- Why SQL at all, when pandas could do this?
--   Because on a real job the ledger lives in a database. Pulling a million
--   rows into memory to group them is the wrong instinct. These queries do the
--   aggregation on the server and bring back only the summary.
--
-- Portability note: this is SQLite so the project runs with zero setup. For
-- SQL Server or PostgreSQL the only changes are the date functions
-- (strftime -> FORMAT / TO_CHAR) and || for string concatenation.
-- ===========================================================================


-- ---------------------------------------------------------------------------
-- 1. Headline P&L  — reconciles against accounting.totals() in the test suite
-- ---------------------------------------------------------------------------
SELECT
    ROUND(SUM(CASE WHEN txn_type = 'Income'  THEN amount ELSE 0 END), 2) AS revenue,
    ROUND(SUM(CASE WHEN txn_type = 'Expense' THEN amount ELSE 0 END), 2) AS expenses,
    ROUND(SUM(CASE WHEN txn_type = 'Income'  THEN amount ELSE -amount END), 2) AS net_profit,
    ROUND(100.0 * SUM(CASE WHEN txn_type = 'Income' THEN amount ELSE -amount END)
          / NULLIF(SUM(CASE WHEN txn_type = 'Income' THEN amount ELSE 0 END), 0), 2)
        AS margin_pct,
    COUNT(*) AS transactions
FROM transactions;


-- ---------------------------------------------------------------------------
-- 2. Monthly P&L
-- ---------------------------------------------------------------------------
SELECT
    strftime('%Y-%m', date)                                           AS month,
    ROUND(SUM(CASE WHEN txn_type='Income'  THEN amount ELSE 0 END), 2) AS revenue,
    ROUND(SUM(CASE WHEN txn_type='Expense' THEN amount ELSE 0 END), 2) AS expenses,
    ROUND(SUM(CASE WHEN txn_type='Income'  THEN amount ELSE -amount END), 2) AS profit
FROM transactions
GROUP BY month
ORDER BY month;


-- ---------------------------------------------------------------------------
-- 3. Expenses by ledger head, with share of total
-- ---------------------------------------------------------------------------
SELECT
    category,
    COUNT(*)                   AS entries,
    ROUND(SUM(amount), 2)      AS total,
    ROUND(AVG(amount), 2)      AS average,
    ROUND(MAX(amount), 2)      AS largest,
    ROUND(100.0 * SUM(amount) /
          (SELECT SUM(amount) FROM transactions WHERE txn_type='Expense'), 2)
        AS pct_of_expenses
FROM transactions
WHERE txn_type = 'Expense'
GROUP BY category
ORDER BY total DESC;


-- ---------------------------------------------------------------------------
-- 4. Month-on-month movement by ledger head
--    This is what "why did my expenses go up?" is really asking.
-- ---------------------------------------------------------------------------
WITH monthly AS (
    SELECT strftime('%Y-%m', date) AS month, category, SUM(amount) AS total
    FROM transactions
    WHERE txn_type = 'Expense'
    GROUP BY month, category
)
SELECT
    m.month,
    m.category,
    ROUND(m.total, 2)                                   AS this_month,
    ROUND(p.total, 2)                                   AS prev_month,
    ROUND(m.total - p.total, 2)                         AS movement,
    ROUND(100.0 * (m.total - p.total) / NULLIF(p.total, 0), 1) AS movement_pct
FROM monthly m
JOIN monthly p
      ON p.category = m.category
     AND p.month = strftime('%Y-%m', date(m.month || '-01', '-1 month'))
ORDER BY m.month DESC, ABS(m.total - p.total) DESC;


-- ---------------------------------------------------------------------------
-- 5. Duplicate-payment candidates
--    Same vendor, same amount, within 5 days. This is the SQL version of the
--    rule layer in accaitics/anomaly.py — a duplicate is definitional, so no
--    model is needed and the result is fully explainable to an auditor.
-- ---------------------------------------------------------------------------
SELECT
    a.transaction_id AS txn_a,
    b.transaction_id AS txn_b,
    a.vendor,
    ROUND(a.amount, 2) AS amount,
    a.date AS date_a,
    b.date AS date_b,
    CAST(julianday(b.date) - julianday(a.date) AS INT) AS days_apart
FROM transactions a
JOIN transactions b
      ON  a.vendor = b.vendor
      AND ROUND(a.amount, 2) = ROUND(b.amount, 2)
      AND a.transaction_id < b.transaction_id
      AND julianday(b.date) - julianday(a.date) BETWEEN 0 AND 5
WHERE a.txn_type = 'Expense'
  AND a.vendor <> ''
ORDER BY a.amount DESC;


-- ---------------------------------------------------------------------------
-- 6. Amounts far above normal FOR THEIR OWN LEDGER HEAD
--    Category-relative, because a large raw-material bill is normal and a
--    large stationery bill is not. SQLite has no MEDIAN, so this uses
--    mean + 3 standard deviations, computed longhand.
-- ---------------------------------------------------------------------------
WITH stats AS (
    SELECT category,
           AVG(amount) AS mean_amt,
           SQRT(AVG(amount * amount) - AVG(amount) * AVG(amount)) AS sd_amt
    FROM transactions
    WHERE txn_type = 'Expense'
    GROUP BY category
)
SELECT
    t.transaction_id,
    t.date,
    t.category,
    t.description,
    ROUND(t.amount, 2)                                      AS amount,
    ROUND(s.mean_amt, 2)                                    AS category_mean,
    ROUND((t.amount - s.mean_amt) / NULLIF(s.sd_amt, 0), 2) AS z_score
FROM transactions t
JOIN stats s ON s.category = t.category
WHERE t.txn_type = 'Expense'
  AND (t.amount - s.mean_amt) / NULLIF(s.sd_amt, 0) > 3
ORDER BY z_score DESC;


-- ---------------------------------------------------------------------------
-- 7. Weekend postings above the category average
-- ---------------------------------------------------------------------------
SELECT
    transaction_id, date,
    CASE CAST(strftime('%w', date) AS INT)
        WHEN 0 THEN 'Sunday' WHEN 6 THEN 'Saturday' END AS day_name,
    category, description, ROUND(amount, 2) AS amount
FROM transactions
WHERE txn_type = 'Expense'
  AND CAST(strftime('%w', date) AS INT) IN (0, 6)
  AND amount > (SELECT AVG(amount) FROM transactions t2
                WHERE t2.category = transactions.category AND t2.txn_type='Expense')
ORDER BY amount DESC;


-- ---------------------------------------------------------------------------
-- 8. Vendor spend summary
-- ---------------------------------------------------------------------------
SELECT
    CASE WHEN vendor = '' THEN '(no vendor recorded)' ELSE vendor END AS vendor,
    category,
    COUNT(*)              AS payments,
    ROUND(SUM(amount), 2) AS total_paid,
    ROUND(AVG(amount), 2) AS avg_payment,
    MIN(date)             AS first_payment,
    MAX(date)             AS last_payment
FROM transactions
WHERE txn_type = 'Expense'
GROUP BY vendor, category
ORDER BY total_paid DESC;


-- ---------------------------------------------------------------------------
-- 9. Revenue concentration — how exposed is the business to one customer?
--    Customer is parsed out of the sales-invoice description.
-- ---------------------------------------------------------------------------
SELECT
    TRIM(SUBSTR(description, INSTR(description, ' - ') + 3)) AS customer,
    COUNT(*)              AS invoices,
    ROUND(SUM(amount), 2) AS revenue,
    ROUND(100.0 * SUM(amount) /
          (SELECT SUM(amount) FROM transactions WHERE txn_type='Income'), 2)
        AS pct_of_revenue
FROM transactions
WHERE txn_type = 'Income' AND INSTR(description, ' - ') > 0
GROUP BY customer
ORDER BY revenue DESC
LIMIT 10;


-- ---------------------------------------------------------------------------
-- 10. Payment-method mix — useful for cash-control questions
-- ---------------------------------------------------------------------------
SELECT
    payment_method,
    COUNT(*)              AS entries,
    ROUND(SUM(amount), 2) AS total,
    ROUND(100.0 * COUNT(*) /
          (SELECT COUNT(*) FROM transactions WHERE txn_type='Expense'), 1)
        AS pct_of_entries
FROM transactions
WHERE txn_type = 'Expense'
GROUP BY payment_method
ORDER BY total DESC;
