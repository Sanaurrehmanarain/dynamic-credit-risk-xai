"""Reproducible feature rules for the Lending Club PD cohort."""

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc_log",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal_log",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "acc_now_delinq",
    "pub_rec_bankruptcies",
    "credit_history_months",
    "fico_midpoint",
]

CATEGORICAL_FEATURES = [
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "addr_state",
]

PRICING_NUMERIC_FEATURES = ["int_rate", "installment"]
PRICING_CATEGORICAL_FEATURES = ["grade", "sub_grade"]

BASELINE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PRICING_FEATURES = (
    BASELINE_FEATURES
    + PRICING_NUMERIC_FEATURES
    + PRICING_CATEGORICAL_FEATURES
)

SPLIT_PERIODS = {
    "train": ("2011-01-01", "2013-12-01"),
    "validation": ("2014-01-01", "2014-12-01"),
    "test": ("2015-01-01", "2015-12-01"),
}


def make_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features without fitting on any dataset."""
    result = frame.copy()

    opened = pd.to_datetime(
        result["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce",
    )
    if opened.isna().any():
        raise ValueError("Unparseable earliest_cr_line values found.")
    if opened.gt(result["issue_month"]).any():
        raise ValueError("Credit-line dates after issue_month found.")

    result["credit_history_months"] = (
        (result["issue_month"].dt.year - opened.dt.year) * 12
        + (result["issue_month"].dt.month - opened.dt.month)
    )

    result["emp_length"] = (
        result["emp_length"].astype("string").fillna("Missing")
    )

    result["dti"] = result["dti"].mask(result["dti"].eq(999))
    result["annual_inc_log"] = np.log1p(result["annual_inc"])
    result["revol_bal_log"] = np.log1p(result["revol_bal"])
    result["fico_midpoint"] = (
        result["fico_range_low"] + result["fico_range_high"]
    ) / 2

    return result.drop(columns=["earliest_cr_line"])


def split_by_issue_month(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return non-overlapping, chronological study periods."""
    if frame["issue_month"].isna().any():
        raise ValueError("Missing issue_month values found.")

    splits = {
        name: frame.loc[
            frame["issue_month"].between(start, end)
        ].copy()
        for name, (start, end) in SPLIT_PERIODS.items()
    }

    if sum(len(part) for part in splits.values()) != len(frame):
        raise ValueError("Some rows are outside the declared split periods.")

    return splits
