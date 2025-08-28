import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import tempfile
import os
import time
from openpyxl import load_workbook

st.set_page_config(page_title="King Salman Park - Optimized Matching", layout="wide")
st.title("📊 King Salman Park - Optimized Batch Matching App")

# -----------------------------
# Reset / Clear Uploaded Files safely
# -----------------------------
keys_to_clear = ["uploaded_files", "df1_small", "df2_small", "tmp_path"]

if st.button("🗑 Clear Uploaded Files / Reset App"):
    cleared = False
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
            cleared = True
    if cleared:
        st.success("✅ Uploaded files and session cleared. You can start fresh.")
        st.experimental_rerun()
    else:
        st.success("✅ App is already clean. You can continue normally.")

# -----------------------------
# Upload Excel Files
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Excel files", type="xlsx", accept_multiple_files=True, key="uploaded_files"
)

if uploaded_files:
    st.subheader("Select files to use for matching")
    selected_files = [f for f in uploaded_files if st.checkbox(f.name, value=True)]
    if len(selected_files) >= 2:
        st.success(f"{len(selected_files)} files selected for matching.")

        # -----------------------------
        # Select Columns to Match
        # -----------------------------
        df1_columns = pd.read_excel(selected_files[0], nrows=0).columns.tolist()
        df2_columns = pd.read_excel(selected_files[1], nrows=0).columns.tolist()

        st.subheader("Step 1: Select column to match")
        match_col1 = st.selectbox(f"Column from {selected_files[0].name}", df1_columns)
        match_col2 = st.selectbox(f"Column from {selected_files[1].name}", df2_columns)

        # -----------------------------
        # Select Additional Columns to Keep
        # -----------------------------
        st.subheader("Step 2: Select additional columns to include in the result")
        include_cols1 = st.multiselect(f"Columns from {selected_files[0].name}", df1_columns)
        include_cols2 = st.multiselect(f"Columns from {selected_files[1].name}", df2_columns)

        # -----------------------------
        # Column warning & dynamic batch size
        # -----------------------------
        total_selected = len(include_cols1) + len(include_cols2)
        if total_selected > 10:
            st.warning(f"⚠️ You selected {total_selected} columns. This may slow down the app.")
        batch_size = 200 if total_selected <= 10 else 50

        # -----------------------------
        # Start Matching
        # -----------------------------
        if st.button("Step 3: Start Matching"):
            if not match_col1 or not match_col2:
                st.warning("⚠️ Please select columns to match.")
            elif not include_cols1 and not include_cols2:
                st.warning("⚠️ Please select at least one additional column to include in the result.")
            else:
                st.info("⏳ Matching in progress (optimized memory + batch writing)...")
                try:
                    # Load only necessary columns
                    df1_small = pd.read_excel(selected_files[0], usecols=[match_col1]+include_cols1)
                    df2_small = pd.read_excel(selected_files[1], usecols=[match_col2]+include_cols2)

                    # Normalize function
                    def normalize(text):
                        if pd.isna(text): return ""
                        text = str(text).lower()
                        text = re.sub(r'[^a-z0-9\s]', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        return text

                    # Precompute token sets
                    df1_small['token_set'] = df1_small[match_col1].apply(normalize).str.split().apply(set)
                    df2_small['norm_match'] = df2_small[match_col2].apply(normalize)

                    # Build token → row index map for fast lookup
                    token_map = defaultdict(set)
                    for idx, tokens in df1_small['token_set'].items():
                        for t in tokens:
                            token_map[t].add(idx)

                    # Temp file for results
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    tmp_path = tmp_file.name
                    tmp_file.close()

                    # Create empty Excel with header
                    pd.DataFrame(columns=include_cols1+include_cols2).to_excel(tmp_path, index=False)

                    total_rows = len(df2_small)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()

                    buffer_rows = []

                    # -----------------------------
                    # Optimized batch matching loop
                    # -----------------------------
                    for idx, row in df2_small.iterrows():
                        mh_tokens = set(row['norm_match'].split())
                        if mh_tokens:
                            candidate_sets = [token_map[t] for t in mh_tokens if t in token_map]
                            if candidate_sets:
                                candidate_indices = set.intersection(*candidate_sets)
                                if candidate_indices:
                                    matched_rows = df1_small.loc[list(candidate_indices), include_cols1].copy()
                                    for col in include_cols2:
                                        matched_rows[col] = row[col]
                                    buffer_rows.append(matched_rows)

                                    if len(buffer_rows) >= batch_size:
                                        batch_df = pd.concat(buffer_rows, ignore_index=True)
                                        with pd.ExcelWriter(tmp_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                                            startrow = writer.sheets['Sheet1'].max_row
                                            batch_df.to_excel(writer, index=False, header=False, startrow=startrow)
                                        buffer_rows = []

                        # Update progress
                        progress_bar.progress((idx+1)/total_rows)
                        status_text.text(f"Processing row {idx+1}/{total_rows} ({(idx+1)/total_rows*100:.1f}%)")

                    # Write remaining rows
                    if buffer_rows:
                        batch_df = pd.concat(buffer_rows, ignore_index=True)
                        with pd.ExcelWriter(tmp_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                            startrow = writer.sheets['Sheet1'].max_row
                            batch_df.to_excel(writer, index=False, header=False, startrow=startrow)

                    end_time = time.time()
                    st.success(f"✅ Matching complete in {end_time - start_time:.2f} seconds")

                    # Preview first 100 rows
                    preview_df = pd.read_excel(tmp_path, nrows=100)
                    st.subheader("Preview of Matched Results (first 100 rows)")
                    st.dataframe(preview_df)

                    # Download full results
                    with open(tmp_path, "rb") as f:
                        st.download_button("💾 Download Full Matched Results", data=f,
                                           file_name="matched_results.xlsx")

                    # Remove temp file
                    os.remove(tmp_path)

                except Exception as e:
                    st.error(f"❌ Error during matching: {e}")

    else:
        st.warning("⚠️ Please select at least 2 files for matching.")
