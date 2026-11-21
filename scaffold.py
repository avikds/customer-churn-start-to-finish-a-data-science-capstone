"""
Customer Churn, Start to Finish: A Data Science Capstone scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Customer churn, start to finish.

Story: take a messy customer export, clean it, find what goes with churn, remove a column that
gives the answer away, split honestly, beat a baseline with a model, then turn the model's scores
into a decision the business can act on.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


def main():
    # ---- 1. Get to know the data and clean it ----
    raw = load_raw_customers()
    audit = audit_table(raw)
    print("raw table:", audit["rows"], "rows,", audit["columns"], "columns,", audit["duplicate_rows"], "duplicate rows")
    print("missing values:", audit["missing"])
    clean = mark_invalid(fix_types(raw))
    print("after cleaning:", len(clean), "customers, churn rate", round(float(clean["churned"].mean()), 3))

    # ---- 2. Explore and prepare ----
    for column in ["contract", "plan"]:
        table = churn_rate_by(clean, column)
        print("churn by", column + ":", {str(k): round(float(v), 3) for k, v in zip(table[column], table["churn_rate"])})
    leaky = find_leaky_columns(clean)
    print("columns that give the answer away:", leaky)
    data = add_features(clean, drop_columns=["customer_id", "signup_date"] + leaky)
    train, test = split_customers(data, test_size=0.25, seed=0)
    print("train:", len(train), "test:", len(test))

    # ---- 3. Model and evaluate honestly ----
    numeric = ["monthly_charge", "age", "support_tickets", "logins_last_30d", "tenure_months", "tickets_per_month"]
    categorical = ["plan", "contract", "payment_method"]
    prep = fit_preprocessor(train, numeric, categorical)
    X_train, names = apply_preprocessor(prep, train)
    X_test, _ = apply_preprocessor(prep, test)
    y_train, y_test = train["churned"].to_numpy(), test["churned"].to_numpy()
    base = baseline_metrics(y_train, y_test)
    print("baseline: accuracy", round(base["accuracy"], 3), "recall", round(base["recall"], 3))
    model = train_model(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    scores = evaluate_model(y_test, p_test)
    print("model at 0.5: auc", round(scores["auc"], 3), "precision", round(scores["precision"], 3),
          "recall", round(scores["recall"], 3))

    # ---- 4. Turn it into a decision ----
    # Choose the threshold on predictions for training rows that each model never saw
    # (5-fold out-of-fold), so the test set is still used only once, here below.
    p_oof = cross_val_predict(LogisticRegression(max_iter=2000, tol=1e-8), X_train, y_train,
                              cv=5, method="predict_proba")[:, 1]
    threshold, _ = choose_threshold(y_train, p_oof, cost_fp=20, cost_fn=60)
    at_best = evaluate_model(y_test, p_test, threshold)
    test_cost = at_best["fp"] * 20 + at_best["fn"] * 60
    print("threshold chosen on training rows:", threshold, "| on the test set: cost", test_cost,
          "recall", round(at_best["recall"], 3), "precision", round(at_best["precision"], 3))
    report = targeting_report(y_test, p_test, top_frac=0.2)
    print("contact the riskiest 20%:", report["contacted"], "customers reach",
          report["churners_reached"], "churners, lift", round(report["lift"], 2))
    print("what moves churn most:")
    for name, coef, direction in top_drivers(model, names, k=5):
        print("  ", name, coef, direction)


if __name__ == "__main__":
    main()

