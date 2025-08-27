import streamlit as st
import pandas as pd
import re
from io import BytesIO
import time

# -----------------------------
# Clear previous session state
# -----------------------------
for key in ['results', 'final_df']:
    if key in st.session_state:
        del st.session_state[key]

st.title('Document Matching App with Progress')

# -----------------------------
# Upload files
# -----------------------------
dc_log_file = st.file_uploader('Upload DC Log Excel file', type='xlsx')
details_file = st.file_uploader('Upload Details Excel file', type='xlsx')

if st.button('Run Matching'):
    if dc_log_file is None or details_file is None:
        st.warning('Please upload both files.')
    else:
        # -----------------------------
        # Read Excel sheets
        # -----------------------------
        dc_log = pd.read_excel(dc_log_file, sheet_name='DC Log')
        details = pd.read_excel(details_file, sheet_name='Details')

        dc_columns = ["DocumentNo.", "Title / Description", "Function", "Discipline",
                      "Document Revision", "(Final Status)\nOpen or Closed"]
        details_columns = ["Package", "Phase", "Part", "Managers", "Service", "Line",
                           "Description", "Start Structure", "End Structure", "MH Type", "Date"]

        dc_log = dc_log[dc_columns]
        details = details[details_columns]

        # -----------------------------
        # Normalize text
        # -----------------------------
        def normalize(text):
            if pd.isna(text): return ""
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        dc_log['norm_title'] = dc_log["Title / Description"].apply(normalize)
        details['norm_mh'] = details["Start Structure"].apply(normalize)
        dc_log['title_tokens'] = dc_log['norm_title'].str.split()

        # -----------------------------
        # Matching loop with live progress
        # -----------------------------
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_rows = len(details)
        for idx, mh_row in enumerate(details.itertuples(index=False), 1):
            norm_mh = mh_row.norm_mh
            if norm_mh:
                mh_tokens = set(norm_mh.split())
                mask = dc_log['title_tokens'].apply(lambda x: mh_tokens.issubset(set(x)))
                matched_rows = dc_log.loc[mask, dc_columns].copy()
                if not matched_rows.empty:
                    for i, col in enumerate(details_columns):
                        matched_rows[col] = mh_row[i]
                    results.append(matched_rows)

            # Update progress and percentage
            progress_bar.progress(idx / total_rows)
            status_text.text(f"Processing row {idx}/{total_rows} ({(idx/total_rows*100):.1f}%)")
            time.sleep(0.01)  # small pause for UI refresh

        # -----------------------------
        # Combine results and display
        # -----------------------------
        final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        if not final_df.empty:
            final_df = final_df[dc_columns + details_columns]
            final_df = final_df.rename(columns={"(Final Status)\nOpen or Closed":"Final Status"})

        st.session_state['final_df'] = final_df

        st.success("Matching complete!")
        st.dataframe(final_df)

        # -----------------------------
        # Download button
        # -----------------------------
        if not final_df.empty:
            output = BytesIO()
            final_df.to_excel(output, index=False)
            st.download_button(
                label='Download Results',
                data=output.getvalue(),
                file_name='matched_results.xlsx'
            )
