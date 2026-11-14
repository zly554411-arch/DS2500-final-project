# Problem Statement:
# investigate the associations between subscription characteristics, 
# particularly plan duration and payment method, 
# and subscription cancellation behavior among KKBOX users

# Component	Variable or method
# Dependent variable: is_cancel
# Main independent variable: payment_plan_days, payment_method_id
# Other independent variables: plan_list_price, actual_amount_paid, payment_method_id
# Exploratory analysis: Bar/Line charts......
# Statistical analysis: Logistic regression (Decision tree or random forest)



import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# 1. LOAD DATA AND CHECK DATA QUALITY
# ==================================================

df = pd.read_csv("kkbox_transactions.csv")

print("Dataset shape:", df.shape)
print("\nColumn information:")
df.info()

print("\nMissing values by column:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==================================================
# 2. DATA PREPROCESSING
# ==================================================

# Convert date columns to datetime format
date_cols = ['transaction_date', 'membership_expire_date']

for col in date_cols:
    df[col] = pd.to_datetime(
        df[col].astype(str),
        format='%Y%m%d',
        errors='coerce'
    )

print("\nMissing or invalid dates after conversion:")
print(df[date_cols].isnull().sum())

print("\nTransaction date range:")
print(df['transaction_date'].min(), "to",
    df['transaction_date'].max())


# Create subscription categories
def classify_plan(days):
    if 28 <= days <= 31:
        return "Monthly"
    elif 360 <= days <= 366:
        return "Annual"
    else:
        return "Other"


df['subscription_type'] = (
    df['payment_plan_days'].apply(classify_plan)
)

plan_order = ['Monthly', 'Annual', 'Other']

print("\nSubscription type distribution:")
print(
    df['subscription_type']
    .value_counts()
    .reindex(plan_order)
)


# ==================================================
# 3. DESCRIPTIVE STATISTICS
# ==================================================

numeric_cols = [
    'payment_plan_days',
    'plan_list_price',
    'actual_amount_paid',
    'is_auto_renew',
    'is_cancel'
]

desc = df[numeric_cols].describe().T

desc['variance'] = df[numeric_cols].var()

print("\nDescriptive statistics:")
print(desc.round(2).to_string())


# Categorical statistics
print("\nPayment method distribution:")
print(
    df['payment_method_id']
    .value_counts()
    .to_string()
)

print("\nSubscription type distribution (%):")
print(
    (df['subscription_type']
    .value_counts(normalize=True)
    .reindex(plan_order) * 100)
    .round(2)
)

print("\nAutomatic renewal counts:")
print(df['is_auto_renew'].value_counts())

print("\nOverall automatic renewal rate:")
print(round(df['is_auto_renew'].mean() * 100, 2), "%")

print("\nCancellation counts:")
print(df['is_cancel'].value_counts())

print("\nOverall cancellation rate:")
print(round(df['is_cancel'].mean() * 100, 2), "%")


# ==================================================
# 4. SINGLE VARIABLE VISUALIZATIONS
# ==================================================

# Chart 1: Subscription type distribution

plan_counts = (
    df['subscription_type']
    .value_counts()
    .reindex(plan_order)
)

plt.figure(figsize=(8, 5))

plt.bar(
    plan_counts.index,
    plan_counts.values
)

plt.title("Distribution of Subscription Types")
plt.xlabel("Subscription Type")
plt.ylabel("Number of Transactions")

plt.savefig("Distribution of Subscription Types")
plt.tight_layout()
plt.show()


# Chart 2: Payment method distribution

payment_counts = (
    df['payment_method_id']
    .value_counts()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))

plt.bar(
    payment_counts.index.astype(str),
    payment_counts.values
)

plt.title("Distribution of Payment Methods")
plt.xlabel("Payment Method ID")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)

plt.savefig("Distribution of Payment Methods")
plt.tight_layout()
plt.show()


# Chart 3: Most common subscription durations

plan_days_counts = (
    df['payment_plan_days']
    .value_counts()
    .head(10)
)

plt.figure(figsize=(9, 5))

plt.bar(
    plan_days_counts.index.astype(str),
    plan_days_counts.values
)

plt.title("Top 10 Most Common Subscription Durations")
plt.xlabel("Subscription Duration (Days)")
plt.ylabel("Number of Transactions")

plt.savefig("Top 10 Most Common Subscription Durations")
plt.tight_layout()
plt.show()


# Chart 4: Distribution of actual amount paid

plt.figure(figsize=(9, 5))

plt.hist(
    df['actual_amount_paid'].dropna(),
    bins=40,
    edgecolor='white'
)

plt.title("Distribution of Actual Amount Paid")
plt.xlabel("Actual Amount Paid")
plt.ylabel("Number of Transactions")

plt.savefig("Distribution of Actual Amount Paid")
plt.tight_layout()
plt.show()
