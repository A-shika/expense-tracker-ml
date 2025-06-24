import streamlit as st
import pandas as pd
import joblib
import os
from datetime import date
import altair as alt

# Load model
model = joblib.load("model/expense_model.pkl")

# CSV file to store data
csv_file = "data/expense.csv"

# Create CSV if it doesn't exist
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Date", "Description", "Amount", "Predicted Category"])
    df.to_csv(csv_file, index=False)

# Streamlit app
st.title("🧾 Personal Expense Tracker")

# Input fields
with st.form("expense_form"):
    description = st.text_input("Enter expense description (all in small)")
    amount = st.number_input("Enter amount", min_value=0.0, format="%.2f")
    expense_date = st.date_input("Select date", value=date.today())
    submit = st.form_submit_button("Add Expense")

if submit:
    if description.strip() == "":
        st.warning("Please enter a description.")
    else:
        description = description.lower().strip()
        category = model.predict([description])[0]
        new_data = pd.DataFrame([[expense_date, description, amount, category]],
                                columns=["Date", "Description", "Amount", "Predicted Category"])
        new_data.to_csv(csv_file, mode='a', header=False, index=False)
        st.success(f"Expense added under '{category}' category.")

# Load data
df = pd.read_csv(csv_file, on_bad_lines='skip')

# Detect columns
category_col = next((col for col in df.columns if col.strip().lower() in ["predicted category", "category"]), None)
amount_col = next((col for col in df.columns if col.strip().lower() == "amount"), None)

if category_col is None or amount_col is None:
    st.warning("⚠️ Required columns ('category' and/or 'amount') are missing.")
else:
    st.subheader("🔍 Filter Your Expenses")

    # Category filter
    categories = sorted(df[category_col].dropna().unique().tolist())
    selected_cat = st.multiselect("Filter by category", categories, default=categories)

    # Date filter
    date_range = st.date_input("Filter by date range", [])
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df["Date"] >= str(start_date)) & (df["Date"] <= str(end_date))]

    # Apply category filter
    df = df[df[category_col].isin(selected_cat)]

    # Expense history
    st.subheader("📊 Expense History")
    st.dataframe(df)

    # Spending chart
    st.subheader("💸 Total Spending by Category")
    try:
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
        df_chart = df.dropna(subset=[amount_col])
        chart = alt.Chart(df_chart).mark_bar().encode(
            x=category_col,
            y=amount_col,
            tooltip=[category_col, amount_col]
        ).properties(width=600)
        st.altair_chart(chart)
    except Exception as e:
        st.warning(f"Chart error: {e}")

        # Total expenditure section
    st.subheader("🧾 Total Expenditure Summary")

    try:
        total_expense = df[amount_col].sum()
        st.markdown(f"### 💵 Total Expense: ₹{total_expense:.2f}")

        category_expense = df.groupby(category_col)[amount_col].sum().sort_values(ascending=False)
        st.markdown("### 📂 Category-wise Totals:")
        st.dataframe(category_expense.reset_index().rename(columns={amount_col: "Total Amount"}))
    except Exception as e:
        st.warning(f"Error calculating totals: {e}")


    # Edit category section
    # Detect description column
desc_col = next((col for col in df.columns if col.strip().lower() == "description"), None)

st.subheader("✏️ Edit Predicted Categories")
if not df.empty and desc_col:
    edited = False
    for i in df.index:
        current_desc = df.at[i, desc_col]
        current_cat = df.at[i, category_col]
        st.markdown(f"**{current_desc}** — ₹{df.at[i, amount_col]}")
        new_cat = st.selectbox(
            f"Edit category (row {i})",
            options=categories,
            index=categories.index(current_cat),
            key=f"edit_{i}"
        )
        if new_cat != current_cat:
            df.at[i, category_col] = new_cat
            edited = True

    if edited and st.button("💾 Save Category Changes"):
        df.to_csv(csv_file, index=False)
        st.success("Categories updated successfully. Refresh to see changes.")
elif not desc_col:
    st.warning("⚠️ Could not find 'Description' column in your CSV.")


# Delete entry section
st.subheader("🗑️ Delete Entry by Row Number")
full_df = pd.read_csv(csv_file, on_bad_lines='skip')

if full_df.empty:
    st.info("No entries available to delete.")
else:
    st.dataframe(full_df)
    delete_row = st.number_input("Enter row number to delete", min_value=0, max_value=len(full_df)-1, step=1)
    if st.button("Delete Row"):
        full_df.drop(index=delete_row, inplace=True)
        full_df.to_csv(csv_file, index=False)
        st.success("Entry deleted. Refresh to update.")
