import pandas as pd
import numpy as np

# --- (1) Load your CSV ---
def load_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"[+] Loaded {len(df)} rows.")
        return df
    except Exception as e:
        raise Exception(f"Failed to load CSV: {str(e)}")

# --- (2) Your row evaluation function (dirty / clean) ---
def can_insert_to_db(row):
    # EXAMPLE logic: 
    # (✅ Customize this to your business rules.)
    if row.isnull().any():
        return "NO-GO"
    if "price" in row and (row['price'] < 0):
        return "NO-GO"
    if "quantity" in row and (row['quantity'] <= 0):
        return "NO-GO"
    return "GO"

# --- (3) Compute Bayesian probabilities ---
def compute_bayesian_probs(df):
    statuses = df.apply(can_insert_to_db, axis=1)
    go_count = (statuses == "GO").sum()
    no_go_count = (statuses == "NO-GO").sum()
    total = len(statuses)

    p_go = go_count / total
    p_no_go = no_go_count / total

    print(f"[Bayesian] GO probability: {p_go:.2f}, NO-GO probability: {p_no_go:.2f}")
    return p_go, p_no_go

# --- (4) Compare with LLM prediction ---
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

# --- (5) Full Pipeline ---
def full_pipeline(csv_file_path, llm_go_pred, llm_no_go_pred):
    df = load_csv(csv_file_path)
    p_go_bayes, p_no_go_bayes = compute_bayesian_probs(df)
    compare_with_llm(p_go_bayes, p_no_go_bayes, llm_go_pred, llm_no_go_pred)

# --------------- EXAMPLE USAGE ---------------

if __name__ == "__main__":
    # Path to your uploaded CSV (adapt for your server path if needed)
    csv_path = "/home/azureuser/backend/macces-prod2-run/static/db/mydb/files/your_file.csv"

    # 🚀 Example: you manually pass LLM outputs (0-1 probabilities)
    llm_predicted_go = 0.85
    llm_predicted_no_go = 0.15

    full_pipeline(csv_path, llm_predicted_go, llm_predicted_no_go)
