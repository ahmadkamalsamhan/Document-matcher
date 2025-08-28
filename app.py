import streamlit as st
import pandas as pd
import re
from io import BytesIO
import time

st.set_page_config(page_title="King Salman Park - Efficient Matching", layout="wide")
st.title("📊 King Salman Park - Memory-Efficient Matching")

# -----------------------------
# Upload Excel Files
# -----------------------------
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

        # -----------------------------
        # Select Columns to Match
        # -----------------------------
        st.subheader("Select columns to match")
        match_col1 = st.selectbox(f"Column from {selected_files[0].name}", df1.columns)
        match_col2 = st.selectbox(f"Column from {selected_files[1].name}", df2.columns)

        st.subheader("Select additional columns to include in result")
        include_cols1 = st.multiselect(f"Columns from {selected_files[0].name}", df1.columns)
        include_cols2 = st.multiselect(f"Columns from {selected_files[1].name}", df2.columns)

        if st.button("Start Matching"):
            # Validation
            if not match_col1 or not match_col2:
                st.warning("⚠️ Please select columns to match.")
            elif not include_cols1 and not include_cols2:
                st.warning("⚠️ Please select at least one column to include in the result.")
            else:
                st.info("⏳ Matching in progress...")

                try:
                    def normalize(text):
                        if pd.isna(text): return ""
                        text = str(text).lower()
                        text = re.sub(r'[^a-z0-9\s]', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        return text

                    # Keep only necessary columns
                    df1_small = df1[[match_col1] + include_cols1].copy()
                    df2_small = df2[[match_col2] + include_cols2].copy()

                    # Precompute token sets
                    df1_small['token_set'] = df1_small[match_col1].apply(normalize).str.split().apply(set)
                    df2_small['norm_match'] = df2_small[match_col2].apply(normalize)

                    output = BytesIO()
                    writer = pd.ExcelWriter(output, engine='openpyxl')

                    total_rows = len(df2_small)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()
                    chunk_count = 0

                    for idx, row in df2_small.iterrows():
                        mh_tokens = set(row['norm_match'].split())
                        if mh_tokens:
                            mask = df1_small['token_set'].apply(lambda x: mh_tokens.issubset(x))
                            matched_rows = df1_small.loc[mask, include_cols1].copy()
                            if not matched_rows.empty:
                                for col in include_cols2:
                                    matched_rows[col] = row[col]

                                # Write chunk to Excel (append)
                                matched_rows.to_excel(writer, index=False, header=chunk_count==0,
                                                      startrow=chunk_count*len(matched_rows))
                                chunk_count += 1

                        # Update progress every 10 rows
                        if idx % 10 == 0 or idx == total_rows - 1:
                            progress_bar.progress((idx + 1) / total_rows)
                            status_text.text(f"Processing row {idx+1}/{total_rows} ({(idx+1)/total_rows*100:.1f}%)")

                    writer.save()
                    output.seek(0)
                    end_time = time.time()

                    st.success(f"✅ Matching complete in {end_time - start_time:.2f} seconds")

                    # Preview first 100 rows
                    preview_df = pd.read_excel(output, nrows=100)
                    st.subheader("Preview of Matched Results (first 100 rows)")
                    st.dataframe(preview_df)

                    # Download button
                    st.download_button("💾 Download Full Matched Results",
                                       data=output.getvalue(),
                                       file_name="matched_results.xlsx")

                except Exception as e:
                    st.error(f"❌ Error during matching: {e}")
    else:
        st.warning("⚠️ Please select at least 2 files for matching.")
