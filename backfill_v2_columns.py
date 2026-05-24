"""
backfill_v2_columns.py
======================
One-off script to backfill columns added to ml_nse_trading_predictions in May 2026.

Columns backfilled for all rows WHERE model_name LIKE '%V2%':
  1. medium_confidence  — derived from existing signal_strength
  2. low_confidence     — derived from existing signal_strength
  3. company            — joined from nse_500_hist_data
  4. volume             — joined from nse_500_hist_data

model_version is left NULL for historical records because the training
timestamps are not retroactively recoverable.

Usage:
    python backfill_v2_columns.py            # run all 4 updates
    python backfill_v2_columns.py --dry-run  # show row counts only, no UPDATE
"""

import sys
import os
import argparse
import pyodbc
from dotenv import load_dotenv

load_dotenv()

# ── DB connection ──────────────────────────────────────────────────────────────

def get_db_connection():
    server   = os.getenv('SQL_SERVER',   r'192.168.86.28\MSSQLSERVER01')
    database = os.getenv('SQL_DATABASE', 'stockdata_db')
    username = os.getenv('SQL_USERNAME', 'remote_user')
    password = os.getenv('SQL_PASSWORD', '')
    driver   = os.getenv('SQL_DRIVER',   'ODBC Driver 17 for SQL Server')

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)


# ── Update steps ───────────────────────────────────────────────────────────────

STEPS = [
    {
        "name": "medium_confidence",
        "description": "Set medium_confidence = 1 where signal_strength = 'Medium'",
        "dry_run_sql": """
            SELECT COUNT(*) FROM ml_nse_trading_predictions
            WHERE model_name LIKE '%V2%'
              AND (medium_confidence IS NULL
                   OR medium_confidence <> CASE WHEN signal_strength = 'Medium' THEN 1 ELSE 0 END)
        """,
        "update_sql": """
            UPDATE ml_nse_trading_predictions
            SET medium_confidence = CASE WHEN signal_strength = 'Medium' THEN 1 ELSE 0 END
            WHERE model_name LIKE '%V2%'
        """,
    },
    {
        "name": "low_confidence",
        "description": "Set low_confidence = 1 where signal_strength = 'Low'",
        "dry_run_sql": """
            SELECT COUNT(*) FROM ml_nse_trading_predictions
            WHERE model_name LIKE '%V2%'
              AND (low_confidence IS NULL
                   OR low_confidence <> CASE WHEN signal_strength = 'Low' THEN 1 ELSE 0 END)
        """,
        "update_sql": """
            UPDATE ml_nse_trading_predictions
            SET low_confidence = CASE WHEN signal_strength = 'Low' THEN 1 ELSE 0 END
            WHERE model_name LIKE '%V2%'
        """,
    },
    {
        "name": "company",
        "description": "Populate company from nse_500_hist_data join",
        "dry_run_sql": """
            SELECT COUNT(*) FROM ml_nse_trading_predictions p
            INNER JOIN nse_500_hist_data h
                ON h.ticker = p.ticker AND h.trading_date = p.trading_date
            WHERE p.model_name LIKE '%V2%'
              AND p.company IS NULL
              AND h.company IS NOT NULL
        """,
        "update_sql": """
            UPDATE p
            SET p.company = h.company
            FROM ml_nse_trading_predictions p
            INNER JOIN nse_500_hist_data h
                ON h.ticker = p.ticker AND h.trading_date = p.trading_date
            WHERE p.model_name LIKE '%V2%'
              AND p.company IS NULL
              AND h.company IS NOT NULL
        """,
    },
    {
        "name": "volume",
        "description": "Populate volume from nse_500_hist_data join",
        "dry_run_sql": """
            SELECT COUNT(*) FROM ml_nse_trading_predictions p
            INNER JOIN nse_500_hist_data h
                ON h.ticker = p.ticker AND h.trading_date = p.trading_date
            WHERE p.model_name LIKE '%V2%'
              AND p.volume IS NULL
              AND h.volume IS NOT NULL
              AND h.volume <> '0'
        """,
        "update_sql": """
            UPDATE p
            SET p.volume = CAST(h.volume AS FLOAT)
            FROM ml_nse_trading_predictions p
            INNER JOIN nse_500_hist_data h
                ON h.ticker = p.ticker AND h.trading_date = p.trading_date
            WHERE p.model_name LIKE '%V2%'
              AND p.volume IS NULL
              AND h.volume IS NOT NULL
              AND h.volume <> '0'
        """,
    },
]


def run_backfill(dry_run: bool) -> None:
    print("\n" + "="*70)
    print("backfill_v2_columns.py — V2 Historical Column Backfill")
    print("="*70)
    mode_label = "DRY RUN (no changes)" if dry_run else "LIVE UPDATE"
    print(f"Mode: {mode_label}\n")

    conn = get_db_connection()
    cursor = conn.cursor()

    for step in STEPS:
        name = step["name"]
        print(f"── {name} ({'dry-run' if dry_run else 'update'}) ──────────────────────")
        print(f"   {step['description']}")

        if dry_run:
            cursor.execute(step["dry_run_sql"])
            count = cursor.fetchone()[0]
            print(f"   Rows that would be updated: {count:,}\n")
        else:
            cursor.execute(step["update_sql"])
            rows_affected = cursor.rowcount
            conn.commit()
            print(f"   Rows updated: {rows_affected:,}\n")

    cursor.close()
    conn.close()

    if dry_run:
        print("Dry run complete — no changes made.")
        print("Re-run without --dry-run to apply updates.")
    else:
        print("Backfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill medium_confidence, low_confidence, company, volume "
                    "for historical V2 predictions in ml_nse_trading_predictions."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show affected row counts without making changes",
    )
    args = parser.parse_args()
    run_backfill(dry_run=args.dry_run)
