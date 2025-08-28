import streamlit as st
import pandas as pd
import re
from io import BytesIO
import time

st.set_page_config(page_title="King Salman Park Matching App", layout="wide")
st.title("📊 King Salman Park - Token-Based Matching App")

# -----------------------------
# Step 1 – Upload Files
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Excel files", type="xlsx", accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

    # Show checkboxes to select files
    st.subheader("Select files to use for matching")
    selected_files = []
    for file in uploaded_files:
        if st.checkbox(file.name, value=True):
            selected_files.append(file)

    if len(selected_files) >= 2:
        # Load first two selected files
        df1 = pd.read_excel(selected_files[0])
        df2 = pd.read_excel(selected_files[1])

        st.write(f"Preview – {selected_files[0].name}")
        st.dataframe(df1.head())
        st.write(f"Preview – {selected_files[1].name}")
        st.dataframe(df2.head())

        # -----------------------------
        # Step 2 – Select Columns
        # -----------------------------
        st.subheader("Step 1: Select Column to Match")
        match_col1 = st.selectbox(f"Column to match from {selected_files[0].name}", df1.columns)
        match_col2 = st.selectbox(f"Column to match from {selected_files[1].name}", df2.columns)

        st.subheader("Step 2: Select Columns to Include in Result")
        include_cols1 = st.multiselect(f"Columns from {selected_files[0].name}", df1.columns, default=df1.columns)
        include_cols2 = st.multiselect(f"Columns from {selected_files[1].name}", df2.columns, default=df2.columns)

        if st.button("Start Matching"):
            st.info("⏳ Matching in progress...")

            # Normalize function
            def normalize(text):
                if pd.isna(text):
                    return ""
                text = str(text).lower()
                text = re.sub(r'[^a-z0-9\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            # Normalize match columns
            df1['norm_match'] = df1[match_col1].apply(normalize)
            df1['tokens_match'] = df1['norm_match'].str.split()
            df2['norm_match'] = df2[match_col2].apply(normalize)

            results = []
            total_rows = len(df2)
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            for idx, row in df2.iterrows():
                tokens = set(row['norm_match'].split())
                if not tokens:
                    continue
                mask = df1['tokens_match'].apply(lambda x: tokens.issubset(set(x)))
                matched_rows = df1.loc[mask, include_cols1].copy()
                if not matched_rows.empty:
                    for col in include_cols2:
                        matched_rows[col] = row[col]
                    results.append(matched_rows)

                progress_bar.progress((idx + 1) / total_rows)
                status_text.text(f"Processing row {idx+1}/{total_rows} ({(idx+1)/total_rows*100:.1f}%)")
                time.sleep(0.01)

            end_time = time.time()
            final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

            if not final_df.empty:
                st.success(f"✅ Matching complete in {end_time - start_time:.2f} seconds")
                st.dataframe(final_df)

                # Download button
                output = BytesIO()
                final_df.to_excel(output, index=False)
                st.download_button(
                    "💾 Download Matched Results",
                    data=output.getvalue(),
                    file_name="matched_results.xlsx"
                )
            else:
                st.warning("⚠️ No matches found.")
    else:
        st.warning("⚠️ Please select at least 2 files for matching.")
