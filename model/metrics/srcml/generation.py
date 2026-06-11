import pandas as pd
from pathlib import Path
import subprocess

file_path = "/Volumes/Engineering/Database/HMCorp/java/train.jsonl"
code_dir = "temp"

Path(f"{code_dir}").mkdir(parents=True, exist_ok=True)


def generate_files():
    df = pd.read_json(file_path, lines=True)
    for _, row in df.iterrows():
        index = row["index"]
        if row["label"] == 0:
            ai_code = row["code"]
            human_code = row["contrast"]
        else:
            human_code = row["code"]
            ai_code = row["contrast"]

        human_file_path = f"{code_dir}/{index}_human"
        ai_file_path = f"{code_dir}/{index}_ai"
        with open(f"{human_file_path}.java", "w") as file:
            file.write(human_code)
        with open(f"{ai_file_path}.java", "w") as file:
            file.write(ai_code)

        subprocess.run(
            ["srcml", f"{human_file_path}.java", "-o", f"{human_file_path}.xml"]
        )
        subprocess.run(["srcml", f"{ai_file_path}.java", "-o", f"{ai_file_path}.xml"])

        print(f"Generated for {index}")



if __name__ == "__main__":
    # generate_files()
