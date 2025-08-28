import streamlit as st
import pandas as pd
import re
from io import BytesIO
import time

st.set_page_config(page_title="Token-Based Matching App", layout="wide")
st.title("📊 King Salman Park - Token-Based Matching App")

# -----------------------------
# Step 1 – Upload Files
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload 2 or more Excel files", type="xlsx", accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

    # Load files into a dictionary
    dataframes = {}
    for file in uploaded_files:
        df = pd.read_excel(file)
        dataframes[file.name] = df
        st.subheader(f"Preview – {file.name}")
        st.dataframe(df.head())

    # -----------------------------
    # Step 2 – Select columns for matching
    # -----------------------------
    st.subheader("Select Columns for Matching")
    file1_name = st.selectbox("Select File 1", list(dataframes.keys()))
    file2_name = st.selectbox("Select File 2", list(dataframes.keys()))

    if file1_name != file2_name:
        df1, df2 = dataframes[file1_name], dataframes[file2_name]

        cols1 = st.multiselect(f"Select one or more columns from {file1_name}", df1.columns)
        cols2 = st.multiselect(f"Select one or more columns from {file2_name}", df2.columns)

        if cols1 and cols2 and st.button("Run Matching"):
            st.info("⏳ Matching in progress...")

            # -----------------------------
            # Step 3 – Token-Based Matching for multiple columns
            # -----------------------------
            def normalize(text):
                if pd.isna(text):
                    return ""
                text = str(text).lower()
                text = re.sub(r'[^a-z0-9\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            # Normalize all selected columns
            for col in cols1:
                df1[f"norm_{col}"] = df1[col].apply(normalize)
                df1[f"tokens_{col}"] = df1[f"norm_{col}"].str.split()

            for col in cols2:
                df2[f"norm_{col}"] = df2[col].apply(normalize)

            results = []
            total_rows = len(df2)
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            # Iterate over df2 rows
            for idx, row in enumerate(df2.itertuples(index=False), 1):
                row_matches = pd.DataFrame()

                # Check all combinations of selected columns
                for col1 in cols1:
                    for col2 in cols2:
                        tokens = set(getattr(row, f"norm_{col2}").split())
                        if not tokens:
                            continue
                        mask = df1[f"tokens_{col1}"].apply(lambda x: tokens.issubset(set(x)))
                        matched_rows = df1.loc[mask].copy()
                        if not matched_rows.empty:
                            for i, c in enumerate(df2.columns):
                                matched_rows[c] = row[i]
                            row_matches = pd.concat([row_matches, matched_rows], ignore_index=True)

                if not row_matches.empty:
                    results.append(row_matches)

                # Update progress
                progress_bar.progress(idx / total_rows)
                status_text.text(f"Processing row {idx}/{total_rows} ({idx/total_rows*100:.1f}%)")
                time.sleep(0.01)

            end_time = time.time()
            final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

            if not final_df.empty:
                st.success(f"✅ Matching complete in {end_time - start_time:.2f} seconds")
                st.dataframe(final_df)

                # -----------------------------
                # Step 4 – Optional Filtering
                # -----------------------------
                st.subheader("Filter Matched Results (Optional)")
                filter_col = st.selectbox("Select column to filter", final_df.columns)
                keywords = st.text_input("Enter keywords (comma-separated)")

                if keywords:
                    with st.spinner("Filtering rows..."):
                        terms = [k.strip().lower() for k in keywords.split(",")]
                        filtered_df = final_df[
                            final_df[filter_col].astype(str).str.lower().apply(
                                lambda x: any(term in x for term in terms)
                            )
                        ]
                        st.write(f"Filtered rows: {len(filtered_df)}")
                        st.dataframe(filtered_df)

                # -----------------------------
                # Step 5 – Download Results
                # -----------------------------
                def to_excel(df):
                    output = BytesIO()
                    df.to_excel(output, index=False)
                    return output.getvalue()

                st.download_button(
                    "💾 Download Full Matched Result",
                    data=to_excel(final_df),
                    file_name="matched_results.xlsx"
                )

                if keywords:
                    st.download_button(
                        "💾 Download Filtered Result",
                        data=to_excel(filtered_df),
                        file_name="filtered_results.xlsx"
                    )
            else:
                st.warning("⚠️ No matches found.")
