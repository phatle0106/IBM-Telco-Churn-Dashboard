# IBM Telco Churn Dashboard

Simple Streamlit dashboard for exploring Telco customer churn with data engineering, KPI tracking, and interactive visuals.

## Live App

https://ibm-telco-churn-dashboard.streamlit.app/

## What This Project Shows

- Data cleaning and feature engineering on `telco_churn.csv`
- Churn-focused KPIs (customers, churn rate, ARPU, revenue at risk, average tenure)
- Interactive filters for customer segmentation
- Storytelling visuals to highlight churn drivers and risk segments
- Customer drill-down table with CSV export

## Run Locally

1. Install dependencies:
   - `pip install streamlit pandas plotly`
2. Start the app:
   - `streamlit run app.py`
3. Open the local URL shown in your terminal.

## Project Files

- `app.py` - Streamlit dashboard application
- `telco_churn.csv` - Telco churn dataset

## Data Source & Attribution
The dataset used in this dashboard is the **Telco Customer Churn** dataset, originally provided by IBM. 
It was accessed via [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.