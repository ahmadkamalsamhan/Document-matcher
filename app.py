# ------------------------------
# WIR → ITP Tracker (Batch Processing)
# ------------------------------

import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process, utils, cdist
import io
import numpy as np

st.set_page_config(page_title="WIR → ITP Tracker (Batch)", layout="wide")
st.title("📊 WIR → ITP Activity Tracker (Batch Processing)")

st.header("Step 1: Upload WIR and ITP Logs")

wir_file = st.file_uploader("Upload WIR Log (.xlsx)", type=["xlsx"])
itp_file = st.file_uploader("Upload ITP Log (.xlsx)", type=["xlsx"])
activity_file = st.file_uploader("Upload ITP Activities Log (.xlsx)", type=["xlsx"])
threshold = st.slider("Fuzzy Match Threshold (%)", 70, 100, 90)

# ------------------------------
# Step 1: Split WIR into 4 parts
# ------------------------------
if wir_file and st.button("Split WIR into 4 parts"):
    wir_df = pd.read_excel(wir_file)
    total_rows = len(wir_df)
    split_size = total_rows // 4 + 1
    st.write(f"Total rows: {total_rows}, Split size: {split_size}")

    wir_parts = []
    for i in range(4):
        part_df = wir_df.iloc[i*split_size:(i+1)*split_size]
        wir_parts.append(part_df)

        output = io.BytesIO()
        part_df.to_excel(output, index=False)
        output.seek(0)

        st.download_button(
            label=f"Download WIR Part {i+1}",
            data=output.getvalue(),
            file_name=f"WIR_part{i+1}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    st.success("✅ WIR file split into 4 parts successfully!")

# ------------------------------
# Step 2: Match a WIR part
# ------------------------------
st.header("Step 2: Upload a WIR part to match with ITP")

wir_part_file = st.file_uploader("Upload WIR Part (.xlsx)", type=["xlsx"])
if wir_part_file and itp_file and activity_file and st.button("Start Matching Part"):
    st.info("Reading Excel files...")
    wir_part_df = pd.read_excel(wir_part_file)
    itp_df = pd.read_excel(itp_file)
    act_df = pd.read_excel(activity_file)

    # ------------------------------
    # Clean Columns
    # ------------------------------
    def clean_columns(df):
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('\n',' ').str.replace('\r',' ')
        return df

    wir_part_df = clean_columns(wir_part_df)
    itp_df = clean_columns(itp_df)
    act_df = clean_columns(act_df)

    # ------------------------------
    # Detect Columns
    # ------------------------------
    wir_title_col = [c for c in wir_part_df.columns if 'Title / Description2' in c][0]
    act_desc_col = [c for c in act_df.columns if 'Activiy Description' in c][0]
    act_itp_col = [c for c in act_df.columns if 'ITP Reference' in c][0]

    # ------------------------------
    # Normalize WIR
    # ------------------------------
    wir_part_df['ActivitiesList'] = wir_part_df[wir_title_col].astype(str).str.split(r',|\+|/| and |&')
    wir_exp_df = wir_part_df.explode('ActivitiesList').reset_index(drop=True)
    wir_exp_df['ActivityNorm'] = wir_exp_df['ActivitiesList'].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")

    st.success(f"Expanded WIR activities: {len(wir_exp_df)} rows total")

    # ------------------------------
    # Normalize ITP Activities
    # ------------------------------
    act_df['ActivityDescNorm'] = act_df[act_desc_col].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")
    act_df['ITP_Ref_Norm'] = act_df[act_itp_col].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")

    # ------------------------------
    # Vectorized Matching using RapidFuzz cdist
    # ------------------------------
    st.info("Matching WIR activities to ITP activities...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    wir_list = wir_exp_df['ActivityNorm'].tolist()
    itp_list = act_df['ActivityDescNorm'].tolist()

    # Compute similarity matrix
    sim_matrix = cdist(itp_list, wir_list, scorer=fuzz.ratio)

    # Best match per ITP
    best_idx = np.argmax(sim_matrix, axis=1)
    best_score = np.max(sim_matrix, axis=1)

    # Build matches DataFrame
    match_df = pd.DataFrame({
        'ITP Reference': act_df[act_itp_col],
        'Activity Description': act_df[act_desc_col],
        'Best Match WIR Activity': [wir_exp_df['ActivitiesList'][i] for i in best_idx],
        'Score': best_score
    })

    # ------------------------------
    # Filter by threshold
    match_df['Matched'] = match_df['Score'] >= threshold

    st.dataframe(match_df.head(50))

    # ------------------------------
    # Download result for this part
    output = io.BytesIO()
    match_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        label="Download Matched Part Result",
        data=output.getvalue(),
        file_name="WIR_ITP_Matched_Part.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.success("✅ Matching part completed!")

# ------------------------------
# Step 3: Merge all matched parts
# ------------------------------
st.header("Step 3: Merge matched parts")
merge_files = st.file_uploader("Upload all matched parts (.xlsx)", accept_multiple_files=True)
if merge_files and st.button("Merge All Parts"):
    merged_df = pd.DataFrame()
    for f in merge_files:
        df = pd.read_excel(f)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    st.dataframe(merged_df.head(50))

    output = io.BytesIO()
    merged_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        label="Download Final Merged Result",
        data=output.getvalue(),
        file_name="WIR_ITP_Matched_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.success("✅ All parts merged successfully!")
