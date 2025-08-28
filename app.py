# app.py
import streamlit as st
import pandas as pd
import re
import tempfile
import os
import time
from collections import defaultdict, Counter

st.set_page_config(page_title="King Salman Park - Matching & Dashboard", layout="wide")
st.title("📊 King Salman Park — Matching (Part 1) + Dashboard (Part 2)")

# ---------------------------
# Helpers
# ---------------------------
def normalize(text):
    if pd.isna(text): return ""
    s = str(text).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def csv_to_excel(csv_path, xlsx_path, chunksize=20000):
    """Convert CSV -> XLSX in chunks to avoid memory spikes."""
    # Write Excel by reading CSV in chunks and appending
    first = True
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            if first:
                chunk.to_excel(writer, index=False, sheet_name='Sheet1', startrow=0)
                first = False
            else:
                startrow = writer.sheets['Sheet1'].max_row
                chunk.to_excel(writer, index=False, header=False, sheet_name='Sheet1', startrow=startrow)

# ---------------------------
# Top-level Reset Button (safe)
# ---------------------------
keys_to_clear = ["uploaded_files", "matched_csv_path", "matched_xlsx_path"]
if st.button("🗑 Clear Uploaded Files / Reset App"):
    cleared = False
    for k in keys_to_clear:
        if k in st.session_state:
            try:
                del st.session_state[k]
            except Exception:
                pass
            cleared = True
    if cleared:
        st.success("✅ Uploaded files and session cleared. You can start fresh.")
        st.experimental_rerun()
    else:
        st.success("✅ App is already clean. You can continue normally.")

# ---------------------------
# Tabs: Matching / Dashboard
# ---------------------------
tab1, tab2 = st.tabs(["Part 1 — Matching", "Part 2 — Dashboard / Filtering"])

# ---------------------------
# Part 1: Matching (exact Colab logic, memory-safe)
# ---------------------------
with tab1:
    st.header("Part 1 — Non-merged Matching (exact Colab logic)")

    uploaded_files = st.file_uploader(
        "Upload Excel files (.xlsx) — select at least two files", type="xlsx", accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) uploaded. Tick which to use (first two selected will be used).")
        selected_files = [f for f in uploaded_files if st.checkbox(f.name, value=True)]

        if len(selected_files) >= 2:
            st.success(f"{len(selected_files)} files selected for matching.")
            # show first-file columns (read header only)
            df1_cols = pd.read_excel(selected_files[0], nrows=0).columns.tolist()
            df2_cols = pd.read_excel(selected_files[1], nrows=0).columns.tolist()

            st.subheader("Step 1: Choose columns to match (exact subset matching)")
            match_col1 = st.selectbox(f"Match column from {selected_files[0].name}", df1_cols)
            match_col2 = st.selectbox(f"Match column from {selected_files[1].name}", df2_cols)

            st.subheader("Step 2: Choose additional columns to include in the result (from both files)")
            include_cols1 = st.multiselect(f"Columns from {selected_files[0].name} to include", df1_cols)
            include_cols2 = st.multiselect(f"Columns from {selected_files[1].name} to include", df2_cols)

            total_selected_cols = len(include_cols1) + len(include_cols2)
            if total_selected_cols > 12:
                st.warning(f"⚠️ You selected {total_selected_cols} columns. This may slow the app. Consider selecting fewer columns.")

            # Start matching button
            if st.button("Start Matching (exact Colab logic)"):
                # validations
                if not match_col1 or not match_col2:
                    st.warning("Select columns to match.")
                elif not include_cols1 and not include_cols2:
                    st.warning("Select at least one column to include in the result.")
                else:
                    st.info("⏳ Matching in progress (exact logic, memory-safe CSV streaming)...")

                    # Load only needed columns for processing (reduce memory)
                    df1_small = pd.read_excel(selected_files[0], usecols=[match_col1] + include_cols1)
                    df2_small = pd.read_excel(selected_files[1], usecols=[match_col2] + include_cols2)

                    # prepare token sets (like Colab)
                    df1_small['token_set'] = df1_small[match_col1].apply(normalize).str.split().apply(set)
                    df2_small['norm_match'] = df2_small[match_col2].apply(normalize)

                    # Prepare temp CSV for streaming results (fast)
                    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                    tmp_csv_path = tmp_csv.name
                    tmp_csv.close()

                    # write header row (columns = include_cols1 + include_cols2)
                    all_result_cols = include_cols1 + include_cols2
                    pd.DataFrame(columns=all_result_cols).to_csv(tmp_csv_path, index=False)

                    total_rows = len(df2_small)
                    progress = st.progress(0)
                    status = st.empty()
                    start_time = time.time()

                    batch_size = 200
                    buffer = []

                    # Row-by-row matching using exact subset logic (Colab)
                    for idx, (_, mh_row) in enumerate(df2_small.iterrows()):
                        norm_val = mh_row['norm_match']
                        if norm_val:
                            mh_tokens = set(norm_val.split())
                            # exact subset check on df1 token_set
                            mask = df1_small['token_set'].apply(lambda x: mh_tokens.issubset(x))
                            matched_rows = df1_small.loc[mask, include_cols1].copy()

                            if not matched_rows.empty:
                                # add details columns values from mh_row
                                for c in include_cols2:
                                    matched_rows[c] = mh_row[c]
                                buffer.append(matched_rows)

                            # write in batches to CSV (fast I/O)
                            if len(buffer) >= batch_size:
                                batch_df = pd.concat(buffer, ignore_index=True)
                                batch_df.to_csv(tmp_csv_path, mode='a', header=False, index=False)
                                buffer = []

                        # update progress
                        progress.progress((idx + 1) / total_rows)
                        status.text(f"Processing row {idx + 1}/{total_rows} ({(idx + 1)/total_rows*100:.1f}%)")

                    # write remaining buffer
                    if buffer:
                        pd.concat(buffer, ignore_index=True).to_csv(tmp_csv_path, mode='a', header=False, index=False)
                        buffer = []

                    # Convert CSV -> XLSX for download (one-time, chunked)
                    tmp_xlsx = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    tmp_xlsx_path = tmp_xlsx.name
                    tmp_xlsx.close()
                    csv_to_excel(tmp_csv_path, tmp_xlsx_path)

                    # store paths in session for Part 2 usage
                    st.session_state['matched_csv_path'] = tmp_csv_path
                    st.session_state['matched_xlsx_path'] = tmp_xlsx_path

                    end_time = time.time()
                    st.success(f"✅ Matching finished in {end_time - start_time:.1f}s. Results saved for Part 2.")
                    # small preview
                    try:
                        preview_df = pd.read_csv(tmp_csv_path, nrows=100)
                        st.subheader("Preview (first 100 rows) of matched result")
                        st.dataframe(preview_df)
                    except Exception:
                        st.info("Matched file created — preview not available.")

        else:
            st.warning("Please tick/select at least 2 uploaded files to enable matching.")

    else:
        st.info("Upload at least two Excel (.xlsx) files to start Part 1 matching.")

# ---------------------------
# Part 2: Dashboard / Filtering
# ---------------------------
with tab2:
    st.header("Part 2 — Dashboard / Filtering")

    # choose source: use Part1 result if present, else upload a matched file
    use_part1 = 'matched_csv_path' in st.session_state
    uploaded_matched = st.file_uploader("Upload a matched CSV/XLSX (optional) — otherwise use Part 1 result", type=['csv','xlsx'])

    matched_csv_path = None
    matched_xlsx_path = None
    temp_uploaded_csv = None

    if uploaded_matched:
        # if user uploads CSV/XLSX for Part2, save it to a temp CSV (if XLSX convert to CSV)
        up_name = uploaded_matched.name.lower()
        temp_uploaded = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_matched.name)[1])
        temp_uploaded_path = temp_uploaded.name
        with open(temp_uploaded_path, "wb") as f:
            f.write(uploaded_matched.getbuffer())
        temp_uploaded.close()

        if up_name.endswith(".csv"):
            matched_csv_path = temp_uploaded_path
        else:
            # convert uploaded xlsx -> csv for streaming operations
            tmp_conv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp_conv_path = tmp_conv.name
            tmp_conv.close()
            pd.read_excel(temp_uploaded_path).to_csv(tmp_conv_path, index=False)
            matched_csv_path = tmp_conv_path
            # remove original uploaded excel temp file
            os.remove(temp_uploaded_path)
    elif use_part1:
        matched_csv_path = st.session_state.get('matched_csv_path', None)
        matched_xlsx_path = st.session_state.get('matched_xlsx_path', None)
    else:
        st.info("No matched results available. Run Part 1 or upload a matched result file to use the dashboard.")
        matched_csv_path = None

    if matched_csv_path:
        # read header to get columns without loading whole file
        try:
            header_df = pd.read_csv(matched_csv_path, nrows=0)
            available_cols = header_df.columns.tolist()
        except Exception:
            st.error("Unable to read matched result header.")
            available_cols = []

        st.subheader("Select columns to filter (you may pick multiple)")
        filter_cols = st.multiselect("Columns to filter", available_cols)

        # For each filter column, get keywords input
        filter_inputs = {}
        if filter_cols:
            st.info("Enter comma-separated keywords for each selected column (rows must match all column filters).")
            for col in filter_cols:
                v = st.text_input(f"Keywords for '{col}' (comma-separated)", key=f"f_{col}")
                filter_inputs[col] = [k.strip().lower() for k in v.split(",") if k.strip()] if v.strip() else []

            # option: preview only or also full filtered download
            st.subheader("Filter options")
            max_preview = st.number_input("Preview rows (max)", min_value=10, max_value=2000, value=200, step=10)
            out_format = st.radio("Download format for filtered result", ("CSV","XLSX"))

            if st.button("Apply Filters"):
                # build filter plan: only include columns that actually have keywords
                filter_plan = {c: kw for c,kw in filter_inputs.items() if kw}
                if not filter_plan:
                    st.warning("Enter at least one keyword in one of the selected columns.")
                else:
                    # Decide whether to load into memory or stream-chunk filter
                    # Estimate row count: read CSV in chunksize=1 to count? cheaper to read in chunksize=50000 and count
                    row_count = 0
                    try:
                        for _ in pd.read_csv(matched_csv_path, chunksize=50000):
                            row_count += 50000
                    except StopIteration:
                        pass
                    except Exception:
                        # Fallback: try reading small number to estimate; if fails, set large
                        row_count = 1_000_000

                    MEMORY_THRESHOLD = 200_000  # rows - threshold to decide streaming vs in-memory
                    use_streaming = row_count > MEMORY_THRESHOLD

                    st.info(f"Filtering using {'streaming' if use_streaming else 'in-memory'} mode. (Estimated rows: {row_count})")

                    filtered_temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                    filtered_csv_path = filtered_temp_csv.name
                    filtered_temp_csv.close()
                    wrote_header = False

                    # For summary counts (per summary column request)
                    # We'll compute counts for each column if user requests later
                    # Do streaming filter
                    if use_streaming:
                        # streaming: read chunks, filter each chunk, append matches to filtered_csv_path
                        chunksize = 50000
                        total_processed = 0
                        matches_counter = 0
                        # prepare counters for summary if user asks later
                        # Apply chunked filtering
                        for chunk in pd.read_csv(matched_csv_path, chunksize=chunksize):
                            df_chunk = chunk
                            mask = pd.Series([True] * len(df_chunk))
                            for c, kws in filter_plan.items():
                                # build OR regex for this column
                                regex = '|'.join([re.escape(k) for k in kws])
                                # use str.contains (case insensitive)
                                mask = mask & df_chunk[c].astype(str).str.lower().str.contains(regex, na=False)
                            filtered_chunk = df_chunk.loc[mask]
                            if not filtered_chunk.empty:
                                # save header once
                                if not wrote_header:
                                    filtered_chunk.to_csv(filtered_csv_path, index=False, mode='w', header=True)
                                    wrote_header = True
                                else:
                                    filtered_chunk.to_csv(filtered_csv_path, index=False, mode='a', header=False)
                                matches_counter += len(filtered_chunk)
                            total_processed += len(df_chunk)
                            # update progress as fraction of estimated rows (approx)
                            st.progress(min(0.99, total_processed / (row_count if row_count else total_processed)))
                        st.success(f"Filtering complete. {matches_counter} matching rows found (streaming).")
                        # load preview
                        try:
                            preview_df = pd.read_csv(filtered_csv_path, nrows=max_preview)
                        except Exception:
                            preview_df = pd.DataFrame()
                    else:
                        # load entire matched CSV into memory
                        full = pd.read_csv(matched_csv_path)
                        mask = pd.Series([True] * len(full))
                        for c, kws in filter_plan.items():
                            regex = '|'.join([re.escape(k) for k in kws])
                            mask = mask & full[c].astype(str).str.lower().str.contains(regex, na=False)
                        filtered_full = full.loc[mask]
                        filtered_full.to_csv(filtered_csv_path, index=False)
                        st.success(f"Filtering complete. {len(filtered_full)} matching rows found (in-memory).")
                        preview_df = filtered_full.head(max_preview)

                    # show preview
                    if preview_df is not None and not preview_df.empty:
                        st.subheader("Filtered preview")
                        st.dataframe(preview_df)
                    else:
                        st.info("No rows matched or preview is empty.")

                    # Show summary options
                    st.subheader("Summary / Counts")
                    if os.path.exists(filtered_csv_path):
                        # read small sample or compute counts streaming
                        cols_for_summary = st.multiselect("Columns to show counts for", available_cols)
                        if cols_for_summary:
                            summaries = {}
                            # streaming count to avoid loading full filtered file
                            for c in cols_for_summary:
                                counts = Counter()
                                for chunk in pd.read_csv(filtered_csv_path, chunksize=50000):
                                    vals = chunk[c].astype(str).fillna("").tolist()
                                    counts.update(vals)
                                # take top 20
                                top = counts.most_common(20)
                                summaries[c] = top
                                # display table
                                df_sum = pd.DataFrame(top, columns=[c, "Count"])
                                st.write(f"Top values for {c}")
                                st.dataframe(df_sum)
                                # bar chart
                                st.bar_chart(pd.Series(dict(top)))
                    # Download options
                    st.subheader("Download filtered result")
                    if os.path.exists(filtered_csv_path):
                        if out_format == "CSV":
                            with open(filtered_csv_path, "rb") as fh:
                                st.download_button("💾 Download CSV", data=fh, file_name="filtered_results.csv")
                        else:
                            # convert CSV->XLSX on demand (chunked)
                            tmp_xlsx_out = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                            tmp_xlsx_out_path = tmp_xlsx_out.name
                            tmp_xlsx_out.close()
                            csv_to_excel(filtered_csv_path, tmp_xlsx_out_path)
                            with open(tmp_xlsx_out_path, "rb") as fh:
                                st.download_button("💾 Download XLSX", data=fh, file_name="filtered_results.xlsx")
                            os.remove(tmp_xlsx_out_path)
                        # cleanup filtered csv if uploaded_matched was not from Part1 (we used temp file)
                        # keep it until user resets or leaves app
                    else:
                        st.info("No filtered result file to download.")
    else:
        st.info("No matched result available for dashboard. Generate Part 1 results or upload a matched CSV/XLSX file.")

# End of app
