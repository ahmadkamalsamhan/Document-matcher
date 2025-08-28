import streamlit as st
import pandas as pd
import re
from io import BytesIO
import time

st.set_page_config(page_title="King Salman Park - Matching App", layout="wide")
st.title("📊 King Salman Park - Matching App (Fast & Column Selection)")

# Upload Excel files
uploaded_files = st.file_uploader("Upload Excel files", type="xlsx", accept_multiple_files=True)

if uploaded_files:
    selected_files = [f for f in uploaded_files if st.checkbox(f.name, value=True)]
    if len(selected_files) >= 2:
        df1 = pd.read_excel(selected_files[0])
        df2 = pd.read_excel(selected_files[1])

        st.subheader(f"Preview – {selected_files[0].name}")
        st.dataframe(df1.head())
        st.subheader(f"Preview – {selected_files[1].name}")
        st.dataframe(df2.head())

        # Select columns to match
        st.subheader("Select columns to match")
        match_col1 = st.selectbox(f"Column to match from {selected_files[0].name}", df1.columns)
        match_col2 = st.selectbox(f"Column to match from {selected_files[1].name}", df2.columns)

        # Select other columns to include in result
        st.subheader("Select additional columns to include in result")
        include_cols1 = st.multiselect(f"Columns from {selected_files[0].name}", df1.columns, default=df1.columns)
        include_cols2 = st.multiselect(f"Columns from {selected_files[1].name}", df2.columns, default=df2.columns)

        if st.button("Start Matching"):
            st.info("⏳ Matching in progress...")

            # Normalize function
            def normalize(text):
                if pd.isna(text): return ""
                text = str(text).lower()
                text = re.sub(r'[^a-z0-9\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            # Precompute token sets for selected match columns
            df1['token_set'] = df1[match_col1].apply(normalize).str.split().apply(set)
            df2['norm_match'] = df2[match_col2].apply(normalize)

            results = []
            total_rows = len(df2)
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            for idx, row in df2.iterrows():
                mh_tokens = set(row['norm_match'].split())
                if mh_tokens:
                    mask = df1['token_set'].apply(lambda x: mh_tokens.issubset(x))
                    matched_rows = df1.loc[mask, include_cols1].copy()
                    if not matched_rows.empty:
                        for col in include_cols2:
                            matched_rows[col] = row[col]
                        results.append(matched_rows)

                # Update progress every 10 rows
                if idx % 10 == 0 or idx == total_rows - 1:
                    progress_bar.progress((idx+1)/total_rows)
                    status_text.text(f"Processing row {idx+1}/{total_rows} ({(idx+1)/total_rows*100:.1f}%)")

            end_time = time.time()
            final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

            if not final_df.empty:
                st.success(f"✅ Matching complete in {end_time-start_time:.2f} seconds")
                st.dataframe(final_df)

                output = BytesIO()
                final_df.to_excel(output, index=False)
                st.download_button("💾 Download Matched Results", data=output.getvalue(),
                                   file_name="matched_results.xlsx")
            else:
                st.warning("⚠️ No matches found.")
