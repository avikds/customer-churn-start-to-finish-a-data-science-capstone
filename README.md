# Customer Churn, Start to Finish: A Data Science Capstone

Take a subscription company's messy customer export from the first look to a decision the business can act on. Audit and clean the table, find the column that gives the answer away, build features from dates and counts, and split the data once, honestly. Learn the preprocessing from the training rows only, beat a baseline with a logistic regression, and judge it on the test set. Then turn its probabilities into a cost-based threshold and a contact list with its lift, and explain what drives churn.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** audit_table
- [x] **2.** fix_types
- [x] **3.** mark_invalid
- [x] **4.** churn_rate_by
- [x] **5.** find_leaky_columns
- [x] **6.** add_features
- [x] **7.** split_customers
- [x] **8.** fit_preprocessor
- [x] **9.** apply_preprocessor
- [x] **10.** baseline_metrics
- [x] **11.** train_model
- [x] **12.** evaluate_model
- [x] **13.** choose_threshold
- [x] **14.** targeting_report
- [x] **15.** top_drivers

## Results

```
raw table: 2040 rows, 11 columns, 40 duplicate rows
missing values: {'plan': 38, 'monthly_charge': 47, 'age': 72, 'logins_last_30d': 111, 'cancel_survey_score': 1655}
after cleaning: 2000 customers, churn rate 0.231
churn by contract: {'monthly': 0.298, 'annual': 0.1}
churn by plan: {'basic': 0.268, 'premium': 0.209, 'standard': 0.188}
columns that give the answer away: ['cancel_survey_score']
train: 1500 test: 500
baseline: accuracy 0.77 recall 0.0
model at 0.5: auc 0.749 precision 0.735 recall 0.313
threshold chosen on training rows: 0.25 | on the test set: cost 4760 recall 0.626 precision 0.398
contact the riskiest 20%: 100 customers reach 53 churners, lift 2.3
what moves churn most:
   contract=monthly 0.832 raises
   contract=annual -0.832 lowers
   tenure_months -0.656 lowers
   support_tickets 0.585 raises
   logins_last_30d -0.27 lowers
```
