# brew install postgresql@15
# brew services start postgresql@15
# brew services list
# brew link postgresql@15 --force
# psql --version
# postgresql cli: psql postgres 
# brew services stop postgresql@15

# change user name to admin
# psql -U postgres -d postgres
# \dt

import psycopg
import csv
from pathlib import Path
import re
import os
from dotenv import load_dotenv

load_dotenv()

# pip install "psycopg[binary]" python-dotenv

# ---------- CONFIG ----------
# Credentials are read from environment variables (see .env.example).
# Never hardcode real credentials in this file — this repo is public.

DB_CONFIG = {
    "host": os.environ["POSTGRES_HOST"],
    "dbname": os.environ["POSTGRES_DBNAME"],
    "user": os.environ["POSTGRES_USER"],
    "password": os.environ["POSTGRES_PASSWORD"],
    "port": 5432,
    "sslmode": "require",
}

DATA_FOLDER = Path("./data/structured")  # folder containing your CSV datasets


# ---------- HELPERS ----------

def clean_identifier(name: str) -> str:
    """Convert column/table names to safe SQL identifiers."""
    name = name.lower()
    name = re.sub(r"\W+", "_", name)
    return name.strip("_")


def infer_type(value: str):
    """Simple type inference."""
    if value == "":
        return "TEXT"

    try:
        int(value)
        return "INTEGER"
    except:
        pass

    try:
        float(value)
        return "FLOAT"
    except:
        pass

    if value.lower() in {"true", "false", "yes", "no"}:
        return "BOOLEAN"

    return "TEXT"


def infer_schema(csv_path: Path):
    """Infer schema from first non-empty row."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)

    schema = {}

    for col, value in first_row.items():
        schema[clean_identifier(col)] = infer_type(value)

    return schema


# ---------- DATABASE FUNCTIONS ----------

def create_table(conn, table_name, schema):
    columns_sql = ",\n".join(
        f"{col} {dtype}" for col, dtype in schema.items()
    )

    create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            {columns_sql}
        );
    """

    with conn.cursor() as cur:
        cur.execute(create_sql)

    conn.commit()
    print(f"✔ Table created: {table_name}")


def ingest_csv(conn, csv_path, table_name, schema):
    columns = list(schema.keys())

    insert_sql = f"""
        INSERT INTO {table_name} ({",".join(columns)})
        VALUES ({",".join(["%s"] * len(columns))})
    """

    rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            cleaned = []

            for col in row.values():
                if col == "":
                    cleaned.append(None)
                else:
                    cleaned.append(col)

            rows.append(tuple(cleaned))

    with conn.cursor() as cur:
        cur.executemany(insert_sql, rows)

    conn.commit()

    print(f"✔ Loaded {len(rows)} rows into {table_name}")


# ---------- PIPELINE ----------

def process_csv(conn, csv_path):
    table_name = clean_identifier(csv_path.stem)

    print(f"\nProcessing: {csv_path.name}")

    schema = infer_schema(csv_path)

    create_table(conn, table_name, schema)
    ingest_csv(conn, csv_path, table_name, schema)


def main():
    try:
        with psycopg.connect(**DB_CONFIG) as conn:

            csv_files = list(DATA_FOLDER.glob("*.csv"))

            if not csv_files:
                print("No CSV files found.")
                return

            for csv_file in csv_files:
                process_csv(conn, csv_file)

        print("\n✅ All datasets ingested successfully!")

    except Exception as e:
        print("❌ Error:", e)


if __name__ == "__main__":
    main()
