"""
Customer Churn, Start to Finish: A Data Science Capstone

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - audit_table
import numpy as np
import pandas as pd

def load_raw_customers(n=2000, seed=7):
    """A subscription company's customer export on 2025-01-01, as messy as real ones are."""
    rs = np.random.RandomState(seed)
    tenure = rs.randint(1, 61, size=n)
    signup = pd.Timestamp("2025-01-01") - pd.to_timedelta(tenure * 30 + rs.randint(0, 30, size=n), unit="D")
    plan = rs.choice(["basic", "standard", "premium"], size=n, p=[0.5, 0.3, 0.2])
    contract = rs.choice(["monthly", "annual"], size=n, p=[0.65, 0.35])
    base = np.where(plan == "basic", 20.0, np.where(plan == "standard", 45.0, 80.0))
    charge = np.round(base + rs.normal(0, 4, size=n), 2)
    age = rs.randint(18, 75, size=n).astype(float)
    tickets = rs.poisson(1.2, size=n).astype(int)
    logins = rs.poisson(12, size=n).astype(float)
    payment = rs.choice(["card", "bank", "paypal"], size=n, p=[0.6, 0.25, 0.15])
    score = (-0.9 + 1.5 * (contract == "monthly") - 0.04 * tenure + 0.55 * tickets
             - 0.12 * logins + 0.5 * (plan == "basic"))
    churned = rs.rand(n) < 1.0 / (1.0 + np.exp(-score))
    survey = np.where(churned & (rs.rand(n) < 0.8), rs.randint(1, 6, size=n), np.nan)

    def messy(values, share):
        styles = [str.upper, str.title, lambda s: s + " ", lambda s: " " + s]
        pick = rs.randint(0, len(styles), size=n)
        change = rs.rand(n) < share
        return [styles[pick[i]](v) if change[i] else v for i, v in enumerate(values)]

    charge_text = [("$%.2f" % c) if dollar else str(c) for c, dollar in zip(charge, rs.rand(n) < 0.7)]
    df = pd.DataFrame({
        "customer_id": np.arange(10001, 10001 + n),
        "signup_date": signup.strftime("%Y-%m-%d"),
        "plan": messy(plan, 0.25),
        "contract": messy(contract, 0.2),
        "monthly_charge": charge_text,
        "age": age,
        "support_tickets": tickets,
        "logins_last_30d": logins,
        "payment_method": payment,
        "cancel_survey_score": survey,
        "churned": np.where(churned, "Yes", "No"),
    })
    df.loc[rs.rand(n) < 0.03, "monthly_charge"] = None
    df.loc[rs.rand(n) < 0.02, "plan"] = None
    df.loc[rs.rand(n) < 0.04, "age"] = np.nan
    df.loc[rs.rand(n) < 0.01, "age"] = rs.choice([0.0, 199.0])
    df.loc[rs.rand(n) < 0.05, "logins_last_30d"] = np.nan
    df.loc[rs.rand(n) < 0.01, "support_tickets"] = -1
    df = pd.concat([df, df.iloc[rs.choice(n, size=40, replace=False)]], ignore_index=True)
    return df.iloc[rs.permutation(len(df))].reset_index(drop=True)

# ---- end provided


def audit_table(df):
    """
    Generate a summary report of a DataFrame's structure and data quality.

    Returns:
        dict: Contains the number of rows, columns, duplicate rows,
              missing values, and non-numeric columns.
    """

    # Get the total number of rows and columns.
    rows = df.shape[0]
    columns = df.shape[1]

    # Count exact duplicate rows, excluding their first occurrences.
    duplicate_rows = df.duplicated().sum()

    # Count missing values only for columns that contain at least one.
    # The dictionary preserves the original column order.
    missing = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().any()
    }

    # Identify columns whose data types are not numeric.
    # Preserve the original column order.
    non_numeric = [
        col
        for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col].dtype)
    ]

    # Return the complete audit report.
    return {
        "rows": int(rows),
        "columns": int(columns),
        "duplicate_rows": int(duplicate_rows),
        "missing": missing,
        "non_numeric": non_numeric
    }

# Step 2 - fix_types
def fix_types(df):
    """
    Remove exact duplicates and convert columns to usable data types.
    Returns a cleaned copy without modifying the original DataFrame.
    """

    # Remove exact duplicate rows and reset the index.
    out = df.drop_duplicates(keep="first").reset_index(drop=True).copy()

    # Standardize categorical columns by removing surrounding
    # whitespace and converting text to lowercase.
    # Missing values remain missing.
    categorical_cols = ["plan", "contract", "payment_method"]

    for col in categorical_cols:
        out[col] = out[col].str.strip().str.lower()

    # Remove the dollar sign from monthly charges and convert
    # the values to floating-point numbers.
    # Missing values remain missing.
    out["monthly_charge"] = (
        out["monthly_charge"]
        .str.replace("$", "", regex=False)
        .astype(float)
    )

    # Convert signup_date from string to pandas datetime format.
    out["signup_date"] = pd.to_datetime(out["signup_date"])

    # Encode the churn target:
    # Yes = 1 (churned), No = 0 (not churned).
    out["churned"] = out["churned"].map({
        "Yes": 1,
        "No": 0
    }).astype(int)

    return out

# Step 3 - mark_invalid
def mark_invalid(df):
    """
    Replace impossible age values and negative support ticket counts
    with missing values (NaN).

    Returns:
        A cleaned copy of the original DataFrame.
    """

    # Create a copy to preserve the original DataFrame.
    out = df.copy()

    # Ages below 18 or above 100 are considered invalid.
    # Ages exactly 18 and 100 are allowed.
    invalid_age = (out["age"] < 18) | (out["age"] > 100)
    out.loc[invalid_age, "age"] = np.nan

    # Negative support ticket counts are considered invalid.
    out.loc[out["support_tickets"] < 0, "support_tickets"] = np.nan

    return out

# Step 4 - churn_rate_by
def churn_rate_by(df, column):
    """
    Calculate the number of customers and churn rate
    for each unique value of the specified column.

    Results are sorted by churn rate in descending order.
    Ties are sorted alphabetically by column value.
    Missing values in the specified column are excluded.
    """

    # Group customers by the specified column and calculate
    # the total number of customers and average churn rate.
    result = (
        df.groupby(column, dropna=True)
        .agg(
            customers=("churned", "size"),
            churn_rate=("churned", "mean")
        )
        .reset_index()
    )

    # Sort by churn rate (highest first).
    # For equal churn rates, sort by the column value (A to Z).
    result = result.sort_values(
        by=["churn_rate", column],
        ascending=[False, True]
    )

    # Reset the index to start from 0.
    result = result.reset_index(drop=True)

    return result

# Step 5 - find_leaky_columns
def find_leaky_columns(df, target="churned", min_present=20):
    """
    Identify columns that contain information revealing only one
    particular outcome of the target variable.

    A column is considered leaky when:
    1. It contains at least one missing value.
    2. It has at least min_present non-missing values.
    3. All rows where it is non-missing have the same target value.

    Returns:
        list: Names of leaky columns in their original column order.
    """

    # Store the names of columns identified as leaky.
    leaky_columns = []

    # Examine each column while preserving the original column order.
    for col in df.columns:

        # The target column itself cannot be considered a feature.
        if col == target:
            continue

        # Identify rows where the current column has a non-missing value.
        present = df[col].notna()

        # Condition 1: The column must contain at least one missing value.
        if not df[col].isna().any():
            continue

        # Condition 2: The column must have at least min_present
        # non-missing values.
        if present.sum() < min_present:
            continue

        # Extract target values only for rows where the current
        # column contains a non-missing value.
        target_values = df.loc[present, target]

        # Condition 3: All corresponding target values must be
        # present and identical (representing exactly one outcome).
        if target_values.notna().all() and target_values.nunique(dropna=False) == 1:
            leaky_columns.append(col)

    return leaky_columns

# Step 6 - add_features
def add_features(df, drop_columns, snapshot="2025-01-01"):
    """
    Add tenure_months and tickets_per_month features,
    then remove the specified columns.

    Returns:
        A new DataFrame without modifying the original DataFrame.
    """

    # Create a copy to preserve the original DataFrame.
    out = df.copy()

    # Convert the snapshot date into a pandas Timestamp.
    snapshot_date = pd.Timestamp(snapshot)

    # Calculate customer tenure in months.
    # Integer division by 30 converts the number of days into months.
    out["tenure_months"] = (
        (snapshot_date - out["signup_date"]).dt.days // 30
    )

    # Replace zero tenure with 1 when calculating tickets per month
    # to prevent division by zero.
    # Existing missing values remain missing.
    tenure_denominator = out["tenure_months"].replace(0, 1)

    # Calculate the average number of support tickets per month.
    out["tickets_per_month"] = (
        out["support_tickets"] / tenure_denominator
    )

    # Remove all columns specified in drop_columns.
    out = out.drop(columns=drop_columns)

    return out

# Step 7 - split_customers
from sklearn.model_selection import train_test_split


def split_customers(df, test_size=0.25, seed=0):
    """
    Split customer data into training and testing datasets.

    Uses stratified random sampling based on the churned column
    to preserve the proportion of churned and non-churned customers.

    Returns:
        train: Training DataFrame with a reset index.
        test: Testing DataFrame with a reset index.
    """

    # Perform a stratified random train-test split.
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["churned"]
    )

    # Reset the indexes of both datasets, starting from 0.
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test

# Step 8 - fit_preprocessor
def fit_preprocessor(train, numeric, categorical):
    """
    Learn preprocessing statistics from training data only.

    Calculates:
    - Median values for numerical columns.
    - Mean and population standard deviation after median imputation.
    - Sorted unique category levels for categorical columns.

    Returns:
        dict: Preprocessing information required to transform datasets.
    """

    # Initialize dictionaries to store preprocessing statistics.
    medians = {}
    means = {}
    stds = {}
    levels = {}

    # Learn numerical preprocessing statistics from training data.
    for col in numeric:

        # Calculate the median, ignoring missing values.
        # Convert the result to a standard Python float.
        median = float(train[col].median())
        medians[col] = median

        # Fill missing numerical values using the training median.
        filled = train[col].fillna(median)

        # Calculate the mean after median imputation.
        means[col] = float(filled.mean())

        # Calculate the population standard deviation (ddof=0).
        std = float(filled.std(ddof=0))

        # Replace a zero standard deviation with 1.0
        # to prevent division by zero during standardization.
        if std == 0.0:
            std = 1.0

        stds[col] = std

    # Learn categorical levels from training data only.
    for col in categorical:

        # Exclude missing values, identify unique categories,
        # and sort them in ascending order.
        levels[col] = sorted(
            train[col].dropna().unique().tolist()
        )

    # Return all learned preprocessing information.
    return {
        "numeric": list(numeric),
        "categorical": list(categorical),
        "median": medians,
        "mean": means,
        "std": stds,
        "levels": levels
    }

# Step 9 - apply_preprocessor
def apply_preprocessor(prep, df):
    """
    Apply previously learned preprocessing statistics to a DataFrame.

    Numerical columns:
    - Fill missing values using stored training medians.
    - Standardize using stored training means and standard deviations.

    Categorical columns:
    - Perform one-hot encoding using stored training category levels.
    - Unknown and missing categories are encoded as all zeros.

    Returns:
        X: A 2D NumPy array containing floating-point features.
        names: List of feature names corresponding to the columns of X.
    """

    # Store transformed columns and their corresponding feature names.
    features = []
    names = []

    # Process numerical columns in the order specified in prep.
    for col in prep["numeric"]:

        # Fill missing values using the median learned from training data.
        values = df[col].fillna(prep["median"][col]).to_numpy(dtype=float)

        # Standardize using the training mean and standard deviation.
        standardized = (
            (values - prep["mean"][col]) / prep["std"][col]
        )

        # Append the standardized feature and its name.
        features.append(standardized)
        names.append(col)

    # Process categorical columns in the order specified in prep.
    for col in prep["categorical"]:

        # Use only the category levels learned from training data.
        for level in prep["levels"][col]:

            # Create a binary indicator for the current category.
            # Missing values and previously unseen categories become 0.
            encoded = (
                df[col]
                .eq(level)
                .fillna(False)
                .to_numpy(dtype=float)
            )

            # Append the encoded feature.
            features.append(encoded)

            # Name the feature using the column and category level.
            names.append(f"{col}={level}")

    # Combine all features into a 2D NumPy array.
    # Each feature becomes one column.
    if features:
        X = np.column_stack(features).astype(float)
    else:
        # Handle the case where no features are provided.
        X = np.empty((len(df), 0), dtype=float)

    return X, names

# Step 10 - baseline_metrics
def baseline_metrics(y_train, y_test):
    """
    Calculate baseline classification metrics using a majority-class
    prediction strategy.

    The model always predicts the most common class in y_train.
    If both classes are equally common, it predicts class 0.

    Returns:
        dict: Majority class, test accuracy, test recall,
              and training churn base rate.
    """

    # Convert inputs into NumPy arrays for consistent calculations.
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Count the number of churners (1) and non-churners (0)
    # in the training dataset.
    churn_count = np.sum(y_train == 1)
    non_churn_count = np.sum(y_train == 0)

    # Select the majority class.
    # In case of a tie, choose class 0.
    majority_class = int(churn_count > non_churn_count)

    # The baseline always predicts the majority class
    # for every customer in the test dataset.
    predictions = np.full(len(y_test), majority_class)

    # Calculate the proportion of correct predictions.
    accuracy = float(np.mean(predictions == y_test))

    # Identify actual churners in the test dataset.
    actual_churners = (y_test == 1)

    # Calculate recall.
    # Return 0.0 when the test dataset contains no actual churners.
    if np.any(actual_churners):
        recall = float(
            np.mean(predictions[actual_churners] == 1)
        )
    else:
        recall = 0.0

    # Calculate the proportion of churners in the training dataset.
    base_rate = float(np.mean(y_train == 1))

    # Return all baseline metrics.
    return {
        "majority_class": majority_class,
        "accuracy": accuracy,
        "recall": recall,
        "base_rate": base_rate
    }

# Step 11 - train_model
from sklearn.linear_model import LogisticRegression


def train_model(X, y, C=1.0):
    """
    Train a Logistic Regression classification model.

    Parameters:
        X: Training feature matrix.
        y: Training target values.
        C: Inverse regularization strength (default = 1.0).

    Returns:
        A fitted LogisticRegression model.
    """

    # Initialize the Logistic Regression model.
    model = LogisticRegression(
        C=C,
        max_iter=2000,
        tol=1e-8
    )

    # Fit the model using the provided training data.
    model.fit(X, y)

    # Return the trained model.
    return model

# Step 12 - evaluate_model
from sklearn.metrics import roc_auc_score


def evaluate_model(y_true, p, threshold=0.5):
    """
    Evaluate a binary classification model using predicted
    churn probabilities and a specified classification threshold.

    Returns:
        dict: AUC, precision, recall, F1-score,
              true positives, false positives,
              false negatives and true negatives.
    """

    # Convert actual labels and predicted probabilities
    # into NumPy arrays for consistent calculations.
    y_true = np.asarray(y_true).ravel()
    p = np.asarray(p, dtype=float).ravel()

    # Convert predicted probabilities into binary predictions.
    # A customer is predicted to churn when probability >= threshold.
    y_pred = (p >= threshold).astype(int)

    # Calculate the four confusion matrix components.
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    # Calculate precision.
    # Return 0.0 when there are no predicted positive cases.
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    # Calculate recall.
    # Return 0.0 when there are no actual positive cases.
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # Calculate the F1-score.
    # Return 0.0 when precision and recall are both zero.
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Calculate ROC AUC using predicted probabilities,
    # rather than the threshold-based binary predictions.
    auc = float(roc_auc_score(y_true, p))

    # Return all evaluation metrics.
    return {
        "auc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

# Step 13 - choose_threshold (not yet solved)
# TODO: implement

# Step 14 - targeting_report (not yet solved)
# TODO: implement

# Step 15 - top_drivers (not yet solved)
# TODO: implement

