import pandas as pd

def load_texts_from_csv(csv_path, text_column="text"):
    """
    Loads texts from a CSV file given the column name.
    """
    df = pd.read_csv(csv_path)
    texts = df[text_column].dropna().tolist()
    return texts

def save_texts_to_file(texts, txt_path="dataset.txt"):
    """
    Saves a list of texts to a text file, one text per line.
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.strip() + "\n")
    return txt_path