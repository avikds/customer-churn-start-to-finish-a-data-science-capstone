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
