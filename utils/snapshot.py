import os
import joblib


# ─────────────────────────────────────────────
# Snapshots
# ─────────────────────────────────────────────
def save_snapshot(filename: str, model):
    directory = "backup"
    if not os.path.exists(directory):
        os.makedirs(directory)

    joblib.dump(model, f"{directory}/{filename}.pkl")


def load_snapshot(filename: str):
    directory = "backup"
    if not os.path.exists(directory):
        os.makedirs(directory)

    return joblib.load(f"{directory}/{filename}.pkl")
