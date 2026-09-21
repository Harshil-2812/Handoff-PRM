"""
finalize_dataset.py
-------------------
Combines the clean/successful originals with the strong negative corruptions
to create the final PRM dataset.

1. Reads step4_verified.csv to get the positive class (label=1).
2. Reads step5_corruption_results.csv to get the negative class (label=0).
   Drops any row where weak_negative=1.
3. Writes the combined set to final_dataset.csv.
"""

import csv
import sys

STEP4_CSV = "step4_verified.csv"
STEP5_CSV = "step5_corruption_results.csv"
FINAL_CSV = "final_dataset.csv"

# The standard schema for the final classifier
FIELDNAMES = [
    "task_id", "problem", "handoff", "corruption_type", 
    "label", "label_source", "source"
]

def main():
    positives = []
    negatives = []
    
    # 1. Load the positive class (clean handoffs that passed)
    with open(STEP4_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") == "1":
                # Ensure only the necessary fields are kept
                positives.append({k: row.get(k, "") for k in FIELDNAMES})
                
    # 2. Load the negative class (corruptions that broke Agent B)
    with open(STEP5_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("weak_negative") == "0" and row.get("label") == "0":
                negatives.append({k: row.get(k, "") for k in FIELDNAMES})
                
    total_rows = len(positives) + len(negatives)
    
    # 3. Write final dataset
    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(positives)
        writer.writerows(negatives)

    print("=" * 64)
    print("  Final Dataset Compilation")
    print("=" * 64)
    print(f"  Positives (label=1)  : {len(positives):>4} (from {STEP4_CSV})")
    print(f"  Negatives (label=0)  : {len(negatives):>4} (from {STEP5_CSV})")
    print("-" * 64)
    print(f"  Total rows           : {total_rows:>4}")
    print(f"  Output saved to      : {FINAL_CSV}")
    print("=" * 64)

if __name__ == "__main__":
    main()
