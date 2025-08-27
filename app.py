
import streamlit as st
import pandas as pd
import re
from io import BytesIO
from tqdm import tqdm

st.title('Document Matching App')

dc_log_file = st.file_uploader('Upload DC Log Excel file', type='xlsx')
details_file = st.file_uploader('Upload Details Excel file', type='xlsx')

if st.button('Run Matching'):
    if dc_log_file is None or details_file is None:
        st.warning('Please upload both files.')
    else:
        dc_log = pd.read_excel(dc_log_file, sheet_name='DC Log')
        details = pd.read_excel(details_file, sheet_name='Details')

        dc_columns = ["DocumentNo.", "Title / Description", "Function", "Discipline",
                      "Document Revision", "(Final Status)\nOpen or Closed"]
        details_columns = ["Package", "Phase", "Part", "Managers", "Service", "Line",
                           "Description", "Start Structure", "End Structure", "MH Type", "Date"]

        dc_log = dc_log[dc_columns]
        details = details[details_columns]

        def normalize(text):
            if pd.isna(text): return ""
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        dc_log['norm_title'] = dc_log["Title / Description"].apply(normalize)
        details['norm_mh'] = details["Start Structure"].apply(normalize)
        dc_log['title_tokens'] = dc_log['norm_title'].str.split()

        results = []
        for mh_row in tqdm(details.itertuples(index=False), total=len(details), unit="mh"):
            norm_mh = mh_row.norm_mh
            if not norm_mh:
                continue
            mh_tokens = set(norm_mh.split())
            mask = dc_log['title_tokens'].apply(lambda x: mh_tokens.issubset(set(x)))
            matched_rows = dc_log.loc[mask, dc_columns].copy()
            if matched_rows.empty:
                continue
            for i, col in enumerate(details_columns):
                matched_rows[col] = mh_row[i]
            results.append(matched_rows)

        final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        final_df = final_df[dc_columns + details_columns] if not final_df.empty else final_df
        final_df = final_df.rename(columns={"(Final Status)\nOpen or Closed":"Final Status"})

        st.success("Matching complete!")
        st.dataframe(final_df)

        output = BytesIO()
        final_df.to_excel(output, index=False)
        st.download_button('Download Results', output.getvalue(), 'matched_results.xlsx')
