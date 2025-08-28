# -----------------------------
# PART 3 - Merged Matching with Titles Formatted (User-selected columns like Part 1)
# -----------------------------
st.header("🔹 Part 3: Merged Matching with Titles Formatted")

uploaded_files_part3 = st.file_uploader(
    "Upload Excel files for Part 3", type="xlsx", accept_multiple_files=True, key="part3_files"
)

if uploaded_files_part3:
    st.subheader("Select files to use for Part 3")
    selected_files_part3 = [f for f in uploaded_files_part3 if st.checkbox(f.name, value=True)]
    
    if len(selected_files_part3) >= 2:
        st.success(f"{len(selected_files_part3)} files selected for Part 3.")

        # Load columns for user selection
        df1_cols = pd.read_excel(selected_files_part3[0], nrows=0).columns.tolist()
        df2_cols = pd.read_excel(selected_files_part3[1], nrows=0).columns.tolist()

        st.subheader("Step 1: Select columns from files")
        match_col1 = st.selectbox(f"Column to match from {selected_files_part3[0].name}", df1_cols)
        match_col2 = st.selectbox(f"Column to match from {selected_files_part3[1].name}", df2_cols)

        st.subheader("Step 2: Select additional columns to include in the result")
        include_cols1 = st.multiselect(f"Columns from {selected_files_part3[0].name}", df1_cols)
        include_cols2 = st.multiselect(f"Columns from {selected_files_part3[1].name}", df2_cols)

        if st.button("Step 3: Start Merged Matching Part 3"):
            if not match_col1 or not match_col2:
                st.warning("⚠️ Please select columns to match.")
            else:
                st.info("⏳ Matching in progress (memory-safe, like Part 1)...")
                try:
                    # Load selected columns only
                    df1_small = pd.read_excel(selected_files_part3[0], usecols=[match_col1]+include_cols1)
                    df2_small = pd.read_excel(selected_files_part3[1], usecols=[match_col2]+include_cols2)

                    # Normalize function
                    def normalize(text):
                        if pd.isna(text): return ""
                        text = str(text).lower()
                        text = re.sub(r'[^a-z0-9\s]', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        return text

                    df1_small['norm_title'] = df1_small[match_col1].apply(normalize)
                    df1_small['title_tokens'] = df1_small['norm_title'].str.split()
                    df2_small['norm_mh'] = df2_small[match_col2].apply(normalize)

                    results = []
                    total_rows = len(df2_small)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()
                    batch_size = 200
                    buffer_rows = []

                    # Row-by-row matching
                    for idx, row in df2_small.iterrows():
                        norm_val = row['norm_mh']
                        if not norm_val:
                            continue
                        row_tokens = set(norm_val.split())

                        mask = df1_small['title_tokens'].apply(lambda x: row_tokens.issubset(x))
                        matched_rows = df1_small.loc[mask, include_cols1].copy()

                        if not matched_rows.empty:
                            for col in include_cols2:
                                matched_rows[col] = row[col]
                            buffer_rows.append(matched_rows)

                        if len(buffer_rows) >= batch_size:
                            batch_df = pd.concat(buffer_rows, ignore_index=True)
                            results.append(batch_df)
                            buffer_rows = []

                        progress_bar.progress((idx+1)/total_rows)
                        status_text.text(f"Processing row {idx+1}/{total_rows} ({(idx+1)/total_rows*100:.1f}%)")

                    if buffer_rows:
                        batch_df = pd.concat(buffer_rows, ignore_index=True)
                        results.append(batch_df)

                    final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
                    end_time = time.time()
                    st.success(f"✅ Part 3 Matching complete in {end_time - start_time:.2f} seconds")

                    # Preview first 100 rows
                    st.subheader("Preview of Matched Results (first 100 rows)")
                    st.dataframe(final_df.head(100))

                    # Download full results
                    import tempfile, os
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    final_df.to_excel(tmp_file.name, index=False)
                    with open(tmp_file.name, "rb") as f:
                        st.download_button("💾 Download Part 3 Results", data=f,
                                           file_name="matched_merged_titles_formatted.xlsx")
                    os.remove(tmp_file.name)

                except Exception as e:
                    st.error(f"❌ Error during Part 3 matching: {e}")

    else:
        st.warning("⚠️ Please select at least 2 files for Part 3.")
