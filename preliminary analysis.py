# Problem Statement:
#

# Component	Variable or method
# Dependent variable: 
# Main independent variable: payment_plan_days
# Other independent variables: plan_list_price, actual_amount_paid, payment_method_id
# Exploratory analysis: Bar/Line charts......
# Statistical analysis: Logistic regression (Decision tree or random forest)

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("kkbox_transactions.csv")

# 2. Create subscription categories
def classify_plan(days):
    if days == 30:
        return "Monthly"
    elif days in [360, 365]:
        return "Annual"
    else:
        return "Other"

df['subscription_type'] = df['payment_plan_days'].apply(classify_plan)

# 3. Subscription categories distribution
print("Subscription type distribution:")
print(df['subscription_type'].value_counts())

# 4. Overall auto-renewal rate
print("\nOverall auto-renewal rate:")
print(df['is_auto_renew'].mean())

# 5. Auto-renewal rates by subscription type
print("\nAuto-renewal rate by subscription type:")
print(df.groupby('subscription_type')['is_auto_renew'].mean())

# 6. Numbers of record in each category
print("\nAuto-renewal counts by subscription type:")
print(pd.crosstab(
        df['subscription_type'],
        df['is_auto_renew']
    )
)

# visualization
auto_rates = (
    df.groupby('subscription_type')['is_auto_renew']
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

plt.bar(auto_rates.index,auto_rates.values)

plt.title("Automatic Renewal Rate by Subscription Type")
plt.xlabel("Subscription Type")
plt.ylabel("Automatic Renewal Rate")
plt.ylim(0, 1)

plt.show()


# 7. Create a monthly subscription dataset
monthly = df[df['subscription_type'] == 'Monthly'].copy()

print("Total monthly transactions:", len(monthly))

# 8. Auto-renewal by payment method
print("\nAuto-renewal rate by payment method:")
print(
    monthly.groupby('payment_method_id')['is_auto_renew']
    .agg(['count', 'mean'])
    .sort_values('count', ascending=False)
    .head(15)
)

# 9. Price distributions
print("\nMonthly subscription prices:")
print(
    monthly[['plan_list_price', 'actual_amount_paid']]
    .describe()
)

# 10. Cancellation status
print("\nAuto-renewal rate by cancellation status:")
print(
    pd.crosstab(
        monthly['is_cancel'],
        monthly['is_auto_renew'],
        normalize='index'
    )
)

# 11. Cancellation rate by subscription type
print(df['is_cancel'].value_counts())

print("\nCancellation rate:")
print(df['is_cancel'].mean())

print("\nCancellation rate by subscription type:")
print(
    df.groupby('subscription_type')['is_cancel']
    .agg(['count', 'mean'])
)

monthly = df[df['payment_plan_days'] == 30].copy()

cancel_by_method = (
    monthly.groupby('payment_method_id')['is_cancel']
    .agg(['count', 'mean'])
    .sort_values('count', ascending=False)
)

print(cancel_by_method.head(15))