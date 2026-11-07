#!/usr/bin/env python3
import json
import pandas as pd
from collections import Counter

# Path to your JSON file
INPUT_JSON = 'selected_users2_fully_labeled.json'
# Desired output Excel file
OUTPUT_XLSX = 'occupations_summary.xlsx'

def main():
    # 1) Load JSON
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2) Build DataFrame
    records = []
    for user in data:
        records.append({
            'occupation_L1': user.get('occupation_L1', None),
            'occupation_L2': user.get('occupation_L2', None),
            'occupation_L3': user.get('occupation_L3', None),
            'occupation_L4': user.get('occupation_L4', None),
            'interests_hobbies': user.get('interests_hobbies', None),
        })
    df = pd.DataFrame(records)
    
    # 3) Write to Excel
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"Saved extracted data to {OUTPUT_XLSX}")
    
    # 4) Compute and print stats
    for col in df.columns:
        counts = df[col].value_counts(dropna=True)
        unique_vals = counts.shape[0]
        print(f"\n=== Column: {col} ===")
        print(f"Unique values: {unique_vals}")
        print("Top 5 values:")
        print(counts.head(5).to_string())
    
if __name__ == '__main__':
    main()
