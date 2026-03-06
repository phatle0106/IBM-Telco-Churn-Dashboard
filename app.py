import streamlit as st
import pandas as pd
import plotly.express as px

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

COLOR_CHURN_YES = "#d62728"
COLOR_CHURN_NO = "#1f77b4"
COLOR_RATE = "#ff7f0e"


def normalize_yes_no(value):
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true"}:
        return "Yes"
    if text in {"no", "n", "0", "false"}:
        return "No"
    return str(value).strip()


@st.cache_data
def load_data(path: str = "data/telco_churn.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in working_df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    raw_total = working_df["TotalCharges"].astype(str).str.strip()
    blank_totalcharges_count = int(raw_total.eq("").sum())
    coerced_total = pd.to_numeric(raw_total, errors="coerce")
    coercion_to_nan_count = int(coerced_total.isna().sum())

    working_df["TotalCharges"] = coerced_total.fillna(0)
    working_df["MonthlyCharges"] = pd.to_numeric(working_df["MonthlyCharges"], errors="coerce").fillna(0)
    working_df["tenure"] = pd.to_numeric(working_df["tenure"], errors="coerce").fillna(0)

    yes_no_columns = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
    ]
    for col in yes_no_columns:
        working_df[col] = working_df[col].apply(normalize_yes_no)

    service_columns = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in service_columns:
        working_df[col] = (
            working_df[col]
            .astype(str)
            .str.strip()
            .replace({"No internet service": "No", "No phone service": "No"})
            .apply(normalize_yes_no)
        )

    working_df["SeniorCitizen"] = pd.to_numeric(working_df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    working_df["SeniorCitizenLabel"] = working_df["SeniorCitizen"].map({1: "Yes", 0: "No"}).fillna("Unknown")

    working_df["Churn_Flag"] = (working_df["Churn"] == "Yes").astype(int)

    working_df["TenureBand"] = pd.cut(
        working_df["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12m", "13-24m", "25-48m", "49m+"],
        ordered=True,
    )

    working_df["MonthlyChargeBand"] = pd.cut(
        working_df["MonthlyCharges"],
        bins=[-0.01, 35, 70, float("inf")],
        labels=["Low (<$35)", "Medium ($35-$70)", "High (>$70)"],
        ordered=True,
    )

    add_on_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines",
    ]
    working_df["ServiceCount"] = working_df[add_on_columns].eq("Yes").sum(axis=1)

    working_df["ContractRiskRank"] = working_df["Contract"].map(
        {"Month-to-month": 3, "One year": 2, "Two year": 1}
    ).fillna(0).astype(int)

    working_df["AtRiskRevenue"] = working_df["MonthlyCharges"].where(working_df["Churn"] == "Yes", 0)

    null_counts = working_df[REQUIRED_COLUMNS].isna().sum().to_dict()
    unexpected_null_columns = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
    duplicate_customer_ids = int(working_df["customerID"].duplicated().sum())

    working_df.attrs["data_quality"] = {
        "row_count": int(len(working_df)),
        "duplicate_customer_ids": duplicate_customer_ids,
        "blank_totalcharges_raw": blank_totalcharges_count,
        "coercion_to_nan_totalcharges": coercion_to_nan_count,
        "unexpected_null_counts": unexpected_null_columns,
    }

    return working_df


def compute_kpis(df_filtered: pd.DataFrame) -> dict:
    total_customers = int(len(df_filtered))
    if total_customers == 0:
        return {
            "customers": 0,
            "churn_rate": 0.0,
            "arpu": 0.0,
            "revenue_at_risk": 0.0,
            "avg_tenure": 0.0,
        }

    return {
        "customers": total_customers,
        "churn_rate": float(df_filtered["Churn_Flag"].mean() * 100),
        "arpu": float(df_filtered["MonthlyCharges"].mean()),
        "revenue_at_risk": float(df_filtered["AtRiskRevenue"].sum()),
        "avg_tenure": float(df_filtered["tenure"].mean()),
    }


def build_segment_summary(df_filtered: pd.DataFrame) -> pd.DataFrame:
    if df_filtered.empty:
        return pd.DataFrame(
            columns=["Contract", "TenureBand", "Customers", "ChurnRatePct", "RevenueAtRisk"]
        )

    summary = (
        df_filtered.groupby(["Contract", "TenureBand"], observed=True)
        .agg(
            Customers=("customerID", "count"),
            ChurnRatePct=("Churn_Flag", lambda s: s.mean() * 100),
            RevenueAtRisk=("AtRiskRevenue", "sum"),
        )
        .reset_index()
        .sort_values(["Contract", "TenureBand"])
    )
    return summary


def generate_insights(df_filtered: pd.DataFrame) -> list[str]:
    if df_filtered.empty:
        return ["No customers match the current filters. Expand filter selections to generate insights."]

    insights = []

    contract_view = (
        df_filtered.groupby("Contract", observed=True)["Churn_Flag"].mean().mul(100).sort_values(ascending=False)
    )
    if not contract_view.empty:
        top_contract = contract_view.index[0]
        top_contract_rate = contract_view.iloc[0]
        insights.append(
            f"Highest contract churn is in '{top_contract}' at {top_contract_rate:.1f}% - contract strategy is a primary retention lever."
        )

    payment_view = (
        df_filtered.groupby("PaymentMethod", observed=True)["Churn_Flag"].mean().mul(100).sort_values(ascending=False)
    )
    if not payment_view.empty:
        top_payment = payment_view.index[0]
        top_payment_rate = payment_view.iloc[0]
        insights.append(
            f"'{top_payment}' has the highest churn at {top_payment_rate:.1f}% - target this payment cohort with proactive outreach."
        )

    tenure_view = (
        df_filtered.groupby("TenureBand", observed=True)["Churn_Flag"].mean().mul(100).sort_values(ascending=False)
    )
    if not tenure_view.empty:
        top_tenure = tenure_view.index[0]
        top_tenure_rate = tenure_view.iloc[0]
        insights.append(
            f"Tenure band '{top_tenure}' shows the greatest churn ({top_tenure_rate:.1f}%), indicating where onboarding/early-lifecycle interventions matter most."
        )

    tech_support_view = df_filtered.groupby("TechSupport", observed=True)["Churn_Flag"].mean().mul(100)
    if "No" in tech_support_view.index and "Yes" in tech_support_view.index:
        diff = tech_support_view["No"] - tech_support_view["Yes"]
        insights.append(
            f"Customers without Tech Support churn {diff:.1f} percentage points more than those with it - support adoption is a concrete upsell + retention lever."
        )

    revenue_at_risk = float(df_filtered["AtRiskRevenue"].sum())
    insights.append(f"Current filtered monthly revenue at risk is ${revenue_at_risk:,.0f}.")

    return insights[:5]


def render_data_quality_panel(df: pd.DataFrame) -> None:
    quality = df.attrs.get("data_quality", {})
    with st.expander("Data quality diagnostics", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{quality.get('row_count', 0):,}")
        c2.metric("Duplicate customerID", f"{quality.get('duplicate_customer_ids', 0):,}")
        c3.metric("Blank TotalCharges (raw)", f"{quality.get('blank_totalcharges_raw', 0):,}")

        st.caption(
            "TotalCharges coercion to NaN before imputation: "
            f"{quality.get('coercion_to_nan_totalcharges', 0):,}"
        )

        unexpected_nulls = quality.get("unexpected_null_counts", {})
        if unexpected_nulls:
            st.write("Unexpected null counts in required columns:")
            st.dataframe(
                pd.DataFrame(
                    {"column": list(unexpected_nulls.keys()), "null_count": list(unexpected_nulls.values())}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No unexpected nulls found in required columns.")


st.set_page_config(page_title="Telco Churn Story Dashboard", layout="wide")
st.title("Telco Customer Churn: Data Story Dashboard")
st.caption("Data engineering + diagnostics + narrative analytics for churn and revenue risk.")

try:
    raw_df = load_data("data/telco_churn.csv")
    df = prepare_data(raw_df)
except FileNotFoundError:
    st.error("File not found: data/telco_churn.csv. Place the dataset in the 'data' folder.")
    st.stop()
except ValueError as err:
    st.error(str(err))
    st.stop()
except Exception as err:
    st.error(f"Unexpected error while loading data: {err}")
    st.stop()

render_data_quality_panel(df)

st.sidebar.header("Filters")

contract_options = sorted(df["Contract"].dropna().unique())
internet_options = sorted(df["InternetService"].dropna().unique())
payment_options = sorted(df["PaymentMethod"].dropna().unique())
tenure_options = [str(x) for x in df["TenureBand"].dropna().cat.categories]
senior_options = ["Yes", "No"]

selected_contracts = st.sidebar.multiselect("Contract", contract_options, default=contract_options)
selected_internet = st.sidebar.multiselect("Internet Service", internet_options, default=internet_options)
selected_payment = st.sidebar.multiselect("Payment Method", payment_options, default=payment_options)
selected_senior = st.sidebar.multiselect("Senior Citizen", senior_options, default=senior_options)
selected_tenure = st.sidebar.multiselect("Tenure Band", tenure_options, default=tenure_options)

filtered_df = df[
    df["Contract"].isin(selected_contracts)
    & df["InternetService"].isin(selected_internet)
    & df["PaymentMethod"].isin(selected_payment)
    & df["SeniorCitizenLabel"].isin(selected_senior)
    & df["TenureBand"].astype(str).isin(selected_tenure)
].copy()

if filtered_df.empty:
    st.warning("No records match the current filters. Expand one or more filter selections.")
    st.info("Try selecting all tenure bands and all contract types to re-establish baseline context.")
    st.stop()

kpis = compute_kpis(filtered_df)

st.subheader("1) Executive Snapshot")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{kpis['customers']:,}", help="Total customers in current filter view.")
k2.metric("Churn Rate", f"{kpis['churn_rate']:.1f}%", help="Percent of customers with Churn = Yes.")
k3.metric("ARPU", f"${kpis['arpu']:.2f}", help="Average monthly charge in the selected cohort.")
k4.metric("Revenue at Risk", f"${kpis['revenue_at_risk']:,.0f}", help="Sum of monthly charges from churned customers.")
k5.metric("Avg Tenure", f"{kpis['avg_tenure']:.1f} mo", help="Average customer tenure in months.")

st.subheader("2) Churn Drivers")
c1, c2 = st.columns(2)

with c1:
    contract_churn = (
        filtered_df.groupby("Contract", observed=True)["Churn_Flag"]
        .mean()
        .mul(100)
        .reset_index(name="ChurnRatePct")
        .sort_values("ChurnRatePct", ascending=False)
    )
    fig_contract = px.bar(
        contract_churn,
        x="Contract",
        y="ChurnRatePct",
        title="Which contract types drive the highest churn?",
        color_discrete_sequence=[COLOR_RATE],
    )
    fig_contract.update_layout(showlegend=False, yaxis_title="Churn rate (%)")
    st.plotly_chart(fig_contract, use_container_width=True)
    st.caption("What this means: month-to-month exposure typically concentrates churn risk.")

with c2:
    payment_churn = (
        filtered_df.groupby("PaymentMethod", observed=True)["Churn_Flag"]
        .mean()
        .mul(100)
        .reset_index(name="ChurnRatePct")
        .sort_values("ChurnRatePct", ascending=False)
    )
    fig_payment = px.bar(
        payment_churn,
        x="PaymentMethod",
        y="ChurnRatePct",
        title="Does payment behavior correlate with churn risk?",
        color_discrete_sequence=[COLOR_RATE],
    )
    fig_payment.update_layout(showlegend=False, yaxis_title="Churn rate (%)", xaxis_title="Payment method")
    st.plotly_chart(fig_payment, use_container_width=True)
    st.caption("What this means: high-churn payment groups are candidates for billing and engagement interventions.")

c3, c4 = st.columns(2)

with c3:
    tenure_churn = (
        filtered_df.groupby("TenureBand", observed=True)["Churn_Flag"]
        .mean()
        .mul(100)
        .reset_index(name="ChurnRatePct")
    )
    fig_tenure = px.line(
        tenure_churn,
        x="TenureBand",
        y="ChurnRatePct",
        markers=True,
        title="How does churn change across the customer lifecycle?",
    )
    fig_tenure.update_traces(line_color=COLOR_CHURN_YES)
    fig_tenure.update_layout(yaxis_title="Churn rate (%)", xaxis_title="Tenure band")
    st.plotly_chart(fig_tenure, use_container_width=True)
    st.caption("What this means: elevated early-tenure churn points to onboarding and first-year retention opportunities.")

with c4:
    tech_support_churn = (
        filtered_df.groupby("TechSupport", observed=True)["Churn_Flag"]
        .mean()
        .mul(100)
        .reset_index(name="ChurnRatePct")
        .sort_values("ChurnRatePct", ascending=False)
    )
    fig_support = px.bar(
        tech_support_churn,
        x="TechSupport",
        y="ChurnRatePct",
        title="Does service adoption (Tech Support) reduce churn?",
        color="TechSupport",
        color_discrete_map={"Yes": COLOR_CHURN_NO, "No": COLOR_CHURN_YES},
    )
    fig_support.update_layout(yaxis_title="Churn rate (%)", xaxis_title="Tech Support", legend_title_text="")
    st.plotly_chart(fig_support, use_container_width=True)
    st.caption("What this means: a meaningful gap implies support bundling can lower churn.")

st.subheader("3) Revenue-Risk Segmentation")
segment_summary = build_segment_summary(filtered_df)

if segment_summary.empty:
    st.info("No segment summary available for current filters.")
else:
    heatmap_df = segment_summary.pivot(index="Contract", columns="TenureBand", values="ChurnRatePct").fillna(0)
    fig_heatmap = px.imshow(
        heatmap_df,
        aspect="auto",
        color_continuous_scale="OrRd",
        text_auto=".1f",
        title="Where are churn hotspots by contract and tenure?",
    )
    fig_heatmap.update_layout(coloraxis_colorbar_title="Churn %")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption("What this means: prioritize retention where churn rate and customer volume overlap.")

    st.dataframe(
        segment_summary.style.format({"ChurnRatePct": "{:.1f}%", "RevenueAtRisk": "${:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("4) Customer Profile Drill-Down")
drill_columns = [
    "customerID",
    "gender",
    "SeniorCitizenLabel",
    "tenure",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "ServiceCount",
    "Churn",
]
st.write("Examine the first 100 customers in the current filter view:")
st.dataframe(filtered_df[drill_columns][:100], use_container_width=True, hide_index=True)

csv_bytes = filtered_df[drill_columns].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered customer profile (CSV)",
    data=csv_bytes,
    file_name="filtered_telco_churn_profile.csv",
    mime="text/csv",
)

st.subheader("5) Actionable Insights")
insights = generate_insights(filtered_df)
for insight in insights:
    st.markdown(f"- {insight}")
