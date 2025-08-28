import streamlit as st
import pandas as pd
import re
from tqdm import tqdm
import io

# ====================================================
# Utility: Normalization function (same as Colab)
# ====================================================
def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # keep only letters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# ====================================================
# Streamlit App Layout
# ====================================================
st.set_page_config(page_title="Excel Matcher & Filter", layout="wide")
st.title("📊 Excel Processing App")

tab1, tab2 = st.tabs(["🔗 Part 1: Matching", "🔍 Part 2: Search & Filter"])

# ====================================================
# PART 1: Matching Section
# ====================================================
with tab1:
    st.header("🔗 Part 1: Match Two Excel Files")

    file1 = st.file_uploader("Upload File 1 (dc_log.xlsx)", type=["xlsx"], key="f1")
    file2 = st.file_uploader("Upload File 2 (Details.xlsx)", type=["xlsx"], key="f2")

    if file1 and file2:
        df1 = pd.read_excel(file1)
        df2 = pd.read_excel(file2)

        st.success("✅ Files uploaded successfully!")

        # Column selection
        col1 = st.selectbox("Select column from File 1 to match", df1.columns)
        col2 = st.selectbox("Select column from File 2 to match", df2.columns)

        keep_cols1 = st.multiselect("Select additional columns from File 1 to include", df1.columns)
        keep_cols2 = st.multiselect("Select additional columns from File 2 to include", df2.columns)

        if st.button("🚀 Start Matching"):
            progress = st.progress(0)
            results = []

            df1[f"norm_{col1}"] = df1[col1].apply(normalize)
            df1["tokens"] = df1[f"norm_{col1}"].str.split()

            total = len(df2)
            for idx, row in enumerate(df2.itertuples(index=False), 1):
                norm_val = normalize(getattr(row, col2))
                if not norm_val:
                    continue
                tokens = set(norm_val.split())

                mask = df1["tokens"].apply(lambda x: tokens.issubset(set(x)))
                matched = df1.loc[mask, [col1] + keep_cols1].copy()
                if not matched.empty:
                    for c in keep_cols2 + [col2]:
                        matched[c] = getattr(row, c)
                    results.append(matched)

                if idx % 10 == 0 or idx == total:
                    progress.progress(idx / total)

            if results:
                final_df = pd.concat(results, ignore_index=True)
                st.write("✅ Matching complete. Preview:")
                st.dataframe(final_df.head(200))

                buf = io.BytesIO()
                final_df.to_excel(buf, index=False)
                st.download_button("💾 Download Results (XLSX)", data=buf.getvalue(),
                                   file_name="matched_results.xlsx", mime="application/vnd.ms-excel")
            else:
                st.warning("⚠️ No matches found.")


# ====================================================
# PART 2: Search & Filter Section
# ====================================================
with tab2:
    st.header("🔍 Part 2: Search & Filter a Single Excel File")

    file_search = st.file_uploader("Upload Excel file for filtering", type=["xlsx"], key="fs")

    if file_search:
        df_search = pd.read_excel(file_search)
        st.success("✅ File uploaded successfully!")

        # Select columns to filter
        filter_cols = st.multiselect("Select columns to apply search filters", df_search.columns)

        if filter_cols:
            user_inputs = {}
            for col in filter_cols:
                raw_input = st.text_input(f"Keywords for '{col}' (comma-separated)")
                if raw_input:
                    # normalize user input
                    user_inputs[col] = [normalize(word) for word in raw_input.split(",") if word.strip()]

            preview_rows = st.number_input("Preview rows (max)", min_value=10, max_value=2000, value=200)

            if st.button("🔎 Apply Filter"):
                df_filtered = df_search.copy()

                # normalize selected columns in dataframe
                for col in filter_cols:
                    df_filtered[f"norm_{col}"] = df_filtered[col].apply(normalize)

                for col, keywords in user_inputs.items():
                    df_filtered = df_filtered[
                        df_filtered[f"norm_{col}"].apply(lambda x: all(k in x for k in keywords))
                    ]

                if not df_filtered.empty:
                    st.write("✅ Filtering complete. Preview:")
                    st.dataframe(df_filtered.head(preview_rows))

                    format_choice = st.radio("Download format for filtered result", ["CSV", "XLSX"])

                    if format_choice == "CSV":
                        csv = df_filtered.to_csv(index=False).encode("utf-8")
                        st.download_button("💾 Download CSV", data=csv, file_name="filtered_results.csv",
                                           mime="text/csv")
                    else:
                        buf = io.BytesIO()
                        df_filtered.to_excel(buf, index=False)
                        st.download_button("💾 Download XLSX", data=buf.getvalue(),
                                           file_name="filtered_results.xlsx",
                                           mime="application/vnd.ms-excel")
                else:
                    st.warning("⚠️ No rows matched your filters.")
