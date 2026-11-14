"""
Milestone 3 EDA (Zihan): relationship between cancellation (is_cancel) and four plan/price variables:

Usage:
    python eda_plan_price.py kkbox_transactions.csv [--out figures]

Unit of analysis: transaction .

"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TOP_METHODS = 12


def classify_plan(days):
    if 28 <= days <= 31:
        return "Monthly"
    if 360 <= days <= 366:
        return "Annual"
    return "Other"


def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def rate_table(data, by):
    g = data.groupby(by, observed=True)["is_cancel"].agg(n="size", cancels="sum")
    g["rate_%"] = 100 * g["cancels"] / g["n"]
    lo, hi = wilson_ci(g["cancels"], g["n"])
    g["ci_low_%"], g["ci_high_%"] = 100 * lo, 100 * hi
    return g


def chi_square(data, by):
    table = pd.crosstab(data[by], data["is_cancel"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return "n/a (only one group can cancel)"
    chi2, p, dof, _ = chi2_contingency(table)
    v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
    return f"chi2={chi2:,.1f}, dof={dof}, p={p:.3g}, Cramer's V={v:.3f}"


def numeric_summary(data, col):
    a = data.loc[data["is_cancel"] == 0, col]
    b = data.loc[data["is_cancel"] == 1, col]
    _, p_mw = mannwhitneyu(a, b)
    r, _ = pointbiserialr(data["is_cancel"], data[col])
    summary = data.groupby("is_cancel")[col].agg(["count", "mean", "median", "std", "var"])
    return summary.round(2), f"Mann-Whitney p={p_mw:.3g}, point-biserial r={r:.3f}"


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_axisbelow(True)


def bar_with_ci(ax, labels, table, color=BLUE):
    rates = table["rate_%"].values
    err = [rates - table["ci_low_%"].values, table["ci_high_%"].values - rates]
    x = np.arange(len(labels))
    ax.bar(x, rates, color=color, width=0.6, edgecolor="white", linewidth=2)
    ax.errorbar(x, rates, yerr=err, fmt="none", ecolor=MUTED, capsize=3, linewidth=1)
    for xi, r, n in zip(x, rates, table["n"].values):
        ax.text(xi, table["ci_high_%"].values[xi] + 0.3, f"{r:.2f}%\n(n={n:,})",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, labels)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 140)

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    df["subscription_type"] = df["payment_plan_days"].apply(classify_plan)
    df["discount"] = df["plan_list_price"] - df["actual_amount_paid"]
    df["payment_status"] = np.select(
        [df["actual_amount_paid"] == 0, df["discount"] > 0, df["discount"] < 0],
        ["Free (paid 0)", "Discounted", "Paid above list"], default="Full price")
    auto = df[df["is_auto_renew"] == 1]

    section("0. Sample")
    print(f"all transactions: {len(df):,}   cancel rate {100 * df['is_cancel'].mean():.2f}%")
    print(f"auto-renew transactions: {len(auto):,}   cancel rate {100 * auto['is_cancel'].mean():.2f}%")
    print("cancellations among non-auto-renew transactions:",
          int(df.loc[df["is_auto_renew"] == 0, "is_cancel"].sum()))

    section("1. payment_plan_days / subscription_type vs is_cancel")
    t1 = rate_table(df, "subscription_type").reindex(["Monthly", "Annual", "Other"])
    print(t1.round(2).to_string())
    print(chi_square(df, "subscription_type"))
    print("\nauto-renew share by subscription type (%):")
    print((100 * df.groupby("subscription_type")["is_auto_renew"].mean()).round(2).to_string())
    print("\nauto-renew transactions only:")
    print(rate_table(auto, "subscription_type").round(2).to_string())
    print(chi_square(auto, "subscription_type"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bar_with_ci(ax, t1.index, t1)
    style(ax, "Cancellation rate by subscription type", ylabel="Cancellation rate (%)")
    ax.set_ylim(0, t1["ci_high_%"].max() * 1.45)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig1_subscription_type.png"), dpi=200)
    plt.close(fig)

    section("2. payment_method_id vs is_cancel")
    top = df["payment_method_id"].value_counts().head(TOP_METHODS).index
    t2 = rate_table(df[df["payment_method_id"].isin(top)], "payment_method_id")
    t2["auto_renew_%"] = 100 * df.groupby("payment_method_id")["is_auto_renew"].mean().reindex(t2.index)
    t2a = rate_table(auto[auto["payment_method_id"].isin(top)], "payment_method_id")
    t2["rate_auto_only_%"] = t2a["rate_%"].reindex(t2.index)
    t2 = t2.sort_values("n", ascending=False)
    print(t2.round(2).to_string())
    print("all transactions:", chi_square(df, "payment_method_id"))
    print("auto-renew only: ", chi_square(auto, "payment_method_id"))
    monthly149 = auto[(auto["subscription_type"] == "Monthly") & (auto["plan_list_price"] == 149)]
    print(f"monthly, auto-renew, list price 149 (n={len(monthly149):,}):",
          chi_square(monthly149, "payment_method_id"))

    order = t2.sort_values("rate_auto_only_%", na_position="first").index
    y = np.arange(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(y + h / 2, t2.loc[order, "rate_%"], height=h, color=BLUE, label="All transactions",
            edgecolor="white", linewidth=2)
    ax.barh(y - h / 2, t2.loc[order, "rate_auto_only_%"].fillna(0), height=h, color=ORANGE,
            label="Auto-renew transactions only", edgecolor="white", linewidth=2)
    for yi, m in zip(y, order):
        if np.isnan(t2.loc[m, "rate_auto_only_%"]):
            ax.text(0.15, yi, "no auto-renew option -> cannot cancel", va="center",
                    fontsize=8, color=MUTED)
        else:
            ax.text(t2.loc[m, "rate_auto_only_%"] + 0.15, yi - h / 2,
                    f"{t2.loc[m, 'rate_auto_only_%']:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [f"Method {m}  (n={t2.loc[m, 'n']:,})" for m in order])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    style(ax, f"Cancellation rate by payment method (top {TOP_METHODS} by volume)",
          xlabel="Cancellation rate (%)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig2_payment_method.png"), dpi=200)
    plt.close(fig)

    section("3. plan_list_price vs is_cancel")
    s, test = numeric_summary(df, "plan_list_price")
    print("all transactions:\n" + s.to_string() + "\n" + test)
    s, test = numeric_summary(auto, "plan_list_price")
    print("\nauto-renew only:\n" + s.to_string() + "\n" + test)
    common = auto["plan_list_price"].value_counts()
    common = common[common >= 1000].index.sort_values()
    t3 = rate_table(auto[auto["plan_list_price"].isin(common)], "plan_list_price")
    print("\ncancel rate by list price (auto-renew, prices with >= 1,000 transactions):")
    print(t3.round(2).to_string())
    print(chi_square(auto[auto["plan_list_price"].isin(common)], "plan_list_price"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bar_with_ci(ax, [f"{p} NTD" for p in t3.index], t3)
    style(ax, "Cancellation rate by list price (auto-renew transactions)",
          ylabel="Cancellation rate (%)")
    ax.set_ylim(0, t3["ci_high_%"].max() * 1.45)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig3_list_price.png"), dpi=200)
    plt.close(fig)

    section("4. actual_amount_paid vs is_cancel")
    s, test = numeric_summary(df, "actual_amount_paid")
    print("all transactions:\n" + s.to_string() + "\n" + test)
    s, test = numeric_summary(auto, "actual_amount_paid")
    print("\nauto-renew only:\n" + s.to_string() + "\n" + test)
    print(f"\ncorrelation plan_list_price vs actual_amount_paid: "
          f"{df['plan_list_price'].corr(df['actual_amount_paid']):.3f}")
    print(f"share of transactions where paid == list price: "
          f"{100 * (df['discount'] == 0).mean():.2f}%")
    order4 = ["Full price", "Discounted", "Free (paid 0)", "Paid above list"]
    t4 = rate_table(auto, "payment_status").reindex(order4).dropna()
    print("\ncancel rate by payment status (auto-renew):")
    print(t4.round(2).to_string())
    print(chi_square(auto, "payment_status"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bar_with_ci(ax, t4.index, t4)
    style(ax, "Cancellation rate by amount paid vs list price\n(auto-renew transactions)",
          ylabel="Cancellation rate (%)")
    ax.set_ylim(0, t4["ci_high_%"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig4_amount_paid.png"), dpi=200)
    plt.close(fig)

    print(f"\nFigures saved to {args.out}/")


if __name__ == "__main__":
    main()
