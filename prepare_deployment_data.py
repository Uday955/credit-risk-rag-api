import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("deploy_data", exist_ok=True)

print("Loading X_test...")
X_test = pd.read_csv("data/processed/X_test.csv")

def clean_emp_length(value):
    if pd.isna(value):
        return -1
    elif value == '< 1 year':
        return 0
    elif value == '10+ years':
        return 10
    else:
        return int(value.split()[0])

print("Rebuilding state mapping (this takes about a minute)...")
df = pd.read_csv("data/processed/loans_with_target.csv")
df = df.drop(columns=['emp_title', 'title'])
df['emp_length'] = df['emp_length'].apply(clean_emp_length)
df['term'] = df['term'].str.strip().str.replace(' months', '', regex=False).astype(int)

group_b_columns = [
    'avg_cur_bal', 'mo_sin_old_rev_tl_op', 'tot_coll_amt', 'num_actv_bc_tl',
    'total_rev_hi_lim', 'tot_cur_bal', 'bc_util', 'bc_open_to_buy',
    'num_bc_sats', 'num_sats', 'total_bc_limit', 'mort_acc',
    'acc_open_past_24mths', 'total_bal_ex_mort'
]
for col in group_b_columns:
    df[col] = df[col].fillna(df[col].median())

group_c_columns = [
    'pub_rec_bankruptcies', 'revol_util', 'dti', 'collections_12_mths_ex_med',
    'chargeoff_within_12_mths', 'tax_liens'
]
df = df.dropna(subset=group_c_columns)
df = df.dropna()

X = df.drop(columns=['default'])
y = df['default']
_, X_test_fair, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

state_mapping = dict(zip(X_test['addr_state'].round(6), X_test_fair['addr_state']))

# Save the small files
sample = X_test.iloc[:300]
sample.to_csv("deploy_data/X_test_sample.csv", index=False)

with open("deploy_data/state_mapping.json", "w") as f:
    json.dump({str(k): v for k, v in state_mapping.items()}, f)

# Sanity checks
unmapped = [v for v in sample['addr_state'].round(6) if v not in state_mapping]
print("States in mapping:", len(state_mapping))
print("Sample rows whose state could NOT be decoded (should be 0):", len(unmapped))
print("Done. Files saved in deploy_data/")