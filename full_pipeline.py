import pandas as pd
import numpy as np
import os
import sqlite3

def load_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"[+] Loaded {len(df)} rows.")
        return df
    except Exception as e:
        raise Exception(f"Failed to load CSV: {str(e)}")
    
def get_table_and_columns(df, filepath):
    table_name = os.path.splitext(os.path.basename(filepath))[0]
    if df.columns.str.contains('^Unnamed').all():
        df.columns = [f"{str(dtype)}_{i}" for i, dtype in enumerate(df.dtypes)]
    columns = df.columns.tolist()
    types = []
    for dtype in df.dtypes:
        if pd.api.types.is_integer_dtype(dtype):
            types.append("INTEGER")
        elif pd.api.types.is_float_dtype(dtype):
            types.append("REAL")
        else:
            types.append("TEXT")
    return table_name, columns, types

def create_table(conn, table_name, columns, types):
    cursor = conn.cursor()
    cols = ", ".join([f'"{col}" {typ}' for col, typ in zip(columns, types)])
    sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols});'
    cursor.execute(sql)
    conn.commit()

# Persona 1: Nulls, data type errors, column length mismatch
def persona1_check(row, expected_types, expected_len):
    # Null check
    if row.isnull().any():
        return "NO-GO"
    # Data type check
    for idx, (val, typ) in enumerate(zip(row, expected_types)):
        if typ == "INTEGER":
            if not (isinstance(val, (int, np.integer)) or (isinstance(val, float) and val.is_integer())):
                return "NO-GO"
        elif typ == "REAL":
            if not isinstance(val, (float, int, np.floating, np.integer)):
                return "NO-GO"
        elif typ == "TEXT":
            if not isinstance(val, str):
                return "NO-GO"
    # Column length mismatch
    if len(row) != expected_len:
        return "NO-GO"
    return "GO"

# Persona 2: Outliers, duplicates, invalid quantities
def persona2_check(row, df):
    # Outlier check (example: price > 10000 or < 0)
    if "price" in row:
        try:
            price = float(row["price"])
            if price < 0 or price > 10000:
                return "NO-GO"
        except Exception:
            return "NO-GO"
    # Duplicate check
    if df.duplicated().any():
        return "NO-GO"
    # Invalid quantity
    if "quantity" in row:
        try:
            qty = float(row["quantity"])
            if qty <= 0:
                return "NO-GO"
        except Exception:
            return "NO-GO"
    return "GO"

# Persona 3: Bulk insert test (returns True if successful)
def persona3_bulk_insert(go_rows, conn, table_name):
    if go_rows.empty:
        return False
    cursor = conn.cursor()
    cols = ", ".join([f'"{col}"' for col in go_rows.columns])
    vals = ", ".join(["?"] * len(go_rows.columns))
    sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({vals})'
    try:
        cursor.executemany(sql, go_rows.values.tolist())
        conn.commit()
        print("[Persona 3] Bulk insert successful.")
        return True
    except Exception as e:
        print(f"[Persona 3] Bulk insert failed: {e}")
        return False

def compute_bayesian_probs(statuses):
    go_count = (statuses == "GO").sum()
    no_go_count = (statuses == "NO-GO").sum()
    total = len(statuses)
    p_go = go_count / total
    p_no_go = no_go_count / total
    print(f"[Bayesian] GO probability: {p_go:.2f}, NO-GO probability: {p_no_go:.2f}")
    return p_go, p_no_go

def compare_with_llm(p_go_bayes, p_no_go_bayes, llm_go, llm_no_go):
    print("\n[Comparison]")
    print(f"LLM predicted GO: {llm_go:.2f}, NO-GO: {llm_no_go:.2f}")
    print(f"Bayesian predicted GO: {p_go_bayes:.2f}, NO-GO: {p_no_go_bayes:.2f}")
    difference = abs(p_go_bayes - llm_go) + abs(p_no_go_bayes - llm_no_go)
    print(f"Total Difference Score: {difference:.2f}")
    if difference > 0.2:
        print("⚠️ Warning: Significant difference between LLM and Bayesian prediction.")
    else:
        print("✅ LLM and Bayesian probabilities are aligned.")

def full_pipeline(csv_file_path, llm_go_pred, llm_no_go_pred):
    df = load_csv(csv_file_path)
    table_name, columns, types = get_table_and_columns(df, csv_file_path)
    conn = sqlite3.connect("my_database.sqlite")
    create_table(conn, table_name, columns, types)
    expected_len = len(columns)
    # Persona 1
    persona1_status = df.apply(lambda row: persona1_check(row, types, expected_len), axis=1)
    df_p1 = df[persona1_status == "GO"]
    # Persona 2
    persona2_status = df_p1.apply(lambda row: persona2_check(row, df_p1), axis=1)
    go_rows = df_p1[persona2_status == "GO"]
    # Persona 3
    bulk_insert_success = persona3_bulk_insert(go_rows, conn, table_name)
    # Bayesian stats
    total_status = pd.Series(["NO-GO"] * len(df))
    total_status.loc[go_rows.index] = "GO"
    p_go_bayes, p_no_go_bayes = compute_bayesian_probs(total_status)
    compare_with_llm(p_go_bayes, p_no_go_bayes, llm_go_pred, llm_no_go_pred)
    conn.close()

if __name__ == "__main__":
    csv_path = "raw_data.csv"
    llm_predicted_go = 0.85
    llm_predicted_no_go = 0.15
    full_pipeline(csv_path, llm_predicted_go, llm_predicted_no_go)