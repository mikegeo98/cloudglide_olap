import os
import re
from collections import Counter

# Point this at your local folder of TPC-DS .sql files:
SQL_DIR = '/home/mikeg/Documents/benchmark_data/tpcds_parquet/queries/tpcds'

# Regex to find every FROM … clause (lazy up to the next major keyword)
from_re = re.compile(
    r'\bFROM\b\s*(.*?)\s*(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bWITH\b|$)',
    flags=re.IGNORECASE | re.DOTALL
)

def count_tables(from_clause: str) -> int:
    """
    Given the text after a FROM, strip subqueries and count
    comma-separated table entries.
    """
    # Remove any parenthesized subqueries/tables
    cleaned = re.sub(r'\([^()]*\)', '', from_clause)
    # Split on commas
    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
    return len(parts)

results = []
for fname in os.listdir(SQL_DIR):
    if not fname.lower().endswith('.sql'):
        continue
    path = os.path.join(SQL_DIR, fname)
    with open(path, 'r', encoding='utf8') as f:
        sql = f.read()

    # Find every FROM clause (top-level and nested)
    from_clauses = from_re.findall(sql)
    # Sum up all table counts across them
    total_tables = sum(count_tables(clause) for clause in from_clauses)

    results.append((fname, len(from_clauses), total_tables))

# Sort by most tables referenced
results.sort(key=lambda x: x[2], reverse=True)

print(f"{'Query':30s}  {'#FROMs':>6s}  {'#Tables':>7s}")
print("-" * 50)
for fname, n_from, tbls in results:
    print(f"{fname:30s}  {n_from:6d}  {tbls:7d}")
