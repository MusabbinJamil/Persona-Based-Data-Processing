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
        # No headers, use dtype-based names
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

def insert_row_to_db(row, conn, table_name):
    cursor = conn.cursor()
    cols = ", ".join([f'"{col}"' for col in row.index])
    vals = ", ".join(["?"] * len(row))
    sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({vals})'
    try:
        cursor.execute(sql, tuple(row.values))
        conn.commit()
        return True
    except Exception as e:
        print(f"Insert failed: {e}")
        return False

def can_insert_to_db(row):
    if row.isnull().any():
        return "NO-GO"
    if "price" in row and (row['price'] < 0):
        return "NO-GO"
    if "quantity" in row and (row['quantity'] <= 0):
        return "NO-GO"
    return "GO"

def compute_bayesian_probs(df):
    statuses = df.apply(can_insert_to_db, axis=1)
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
    statuses = []
    for _, row in df.iterrows():
        status = can_insert_to_db(row)
        if status == "GO":
            insert_row_to_db(row, conn, table_name)
        statuses.append(status)
    go_count = statuses.count("GO")
    no_go_count = statuses.count("NO-GO")
    total = len(statuses)
    p_go_bayes = go_count / total
    p_no_go_bayes = no_go_count / total
    compare_with_llm(p_go_bayes, p_no_go_bayes, llm_go_pred, llm_no_go_pred)
    conn.close()

if __name__ == "__main__":
    csv_path = "raw_data.csv"
    llm_predicted_go = 0.85
    llm_predicted_no_go = 0.15
    full_pipeline(csv_path, llm_predicted_go, llm_predicted_no_go)