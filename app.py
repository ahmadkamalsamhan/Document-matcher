import streamlit as st
import pandas as pd
import numpy as np
import io
from rapidfuzz import fuzz, process, cdist
import math

st.set_page_config(page_title="WIR → ITP Tracker", layout="wide")
st.title("📊 WIR → ITP Activity Tracker (Split & Match)")

# ------------------------------
# Upload Files
# ------------------------------
st.header("Step 1: Upload Excel Files")

wir_file = st.file_uploader("Upload WIR Log (.xlsx)", type=["xlsx"])
itp_file = st.file_uploader("Upload ITP Log (.xlsx)", type=["xlsx"])
activity_file = st.file_uploader("Upload ITP Activities Log (.xlsx)", type=["xlsx"])

threshold = st.slider("Fuzzy Match Threshold (%)", 70, 100, 90)

# ------------------------------
# Split WIR into 4 Parts
# ------------------------------
if wir_file:
    st.info("Reading WIR Excel file...")
    wir_df = pd.read_excel(wir_file)
    wir_df.columns = wir_df.columns.str.strip().str.replace('\n',' ').str.replace('\r',' ')
    st.success(f"✅ WIR loaded: {len(wir_df)} rows")

    # Split
    num_parts = 4
    part_size = math.ceil(len(wir_df) / num_parts)
    wir_parts = []
    for i in range(num_parts):
        start = i * part_size
        end = min((i+1) * part_size, len(wir_df))
        part_df = wir_df.iloc[start:end].reset_index(drop=True)
        wir_parts.append(part_df)

    st.subheader("Download WIR Parts")
    for idx, part_df in enumerate(wir_parts):
        output = io.BytesIO()
        part_df.to_excel(output, index=False, sheet_name=f'WIR_Part{idx+1}')
        output.seek(0)
        st.download_button(
            label=f"Download WIR Part {idx+1}",
            data=output.getvalue(),
            file_name=f"WIR_part{idx+1}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ------------------------------
# Step 2: Process Matching for a WIR Part
# ------------------------------
st.header("Step 2: Process Matching (Upload a WIR Part)")

wir_part_file = st.file_uploader("Upload WIR Part to Process", type=["xlsx"], key="part")
if wir_part_file and itp_file and activity_file:
    if st.button("Start Matching"):
        st.info("Reading Excel files...")
        wir_part_df = pd.read_excel(wir_part_file)
        itp_df = pd.read_excel(itp_file)
        act_df = pd.read_excel(activity_file)

        # Clean columns
        def clean_columns(df):
            df.columns = df.columns.str.strip().str.replace('\n',' ').str.replace('\r',' ')
            return df

        wir_part_df = clean_columns(wir_part_df)
        itp_df = clean_columns(itp_df)
        act_df = clean_columns(act_df)

        # Detect Columns
        wir_title_col = [c for c in wir_part_df.columns if 'Title / Description' in c][0]
        act_itp_col = [c for c in act_df.columns if 'ITP Reference' in c][0]
        act_desc_col = [c for c in act_df.columns if 'Activiy Description' in c][0]

        # Normalize WIR
        st.info("🔄 Expanding multi-activity WIR rows...")
        wir_part_df['ActivitiesList'] = wir_part_df[wir_title_col].astype(str).str.split(r',|\+|/| and |&')
        wir_exp_df = wir_part_df.explode('ActivitiesList').reset_index(drop=True)
        wir_exp_df['ActivityNorm'] = wir_exp_df['ActivitiesList'].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")
        st.success(f"✅ Expanded WIR activities: {len(wir_exp_df)} rows total")

        # Normalize ITP Activities
        act_df['ITP_Ref_Norm'] = act_df[act_itp_col].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")
        act_df['ActivityDescNorm'] = act_df[act_desc_col].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")

        # ------------------------------
        # Vectorized Matching with progress
        # ------------------------------
        st.info("🔍 Matching WIR activities to ITP activities...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        wir_list = wir_exp_df['ActivityNorm'].tolist()
        itp_list = act_df['ActivityDescNorm'].tolist()

        batch_size = 5000
        best_matches = []
        scores = []

        for start in range(0, len(itp_list), batch_size):
            end = min(start + batch_size, len(itp_list))
            batch_itp = itp_list[start:end]
            sim_matrix = cdist(batch_itp, wir_list, scorer=fuzz.ratio)
            best_idx = np.argmax(sim_matrix, axis=1)
            best_score = np.max(sim_matrix, axis=1)
            best_matches.extend([wir_exp_df['ActivityNorm'][i] for i in best_idx])
            scores.extend(best_score)
            progress_bar.progress(end / len(itp_list))
            status_text.text(f"Processed {end}/{len(itp_list)} ITP activities")

        # Build matches DataFrame
        match_df = pd.DataFrame({
            'ITP Reference': act_df[act_itp_col],
            'Activity Description': act_df[act_desc_col],
            'Best Match': best_matches,
            'Score': scores
        })
        match_df['Status'] = match_df['Score'].apply(lambda x: 1 if x >= threshold else 0)

        st.subheader("Matching Result Preview")
        st.dataframe(match_df.head(50))

        # Excel Output
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            match_df.to_excel(writer, index=False, sheet_name='Matches')
        output.seek(0)

        st.download_button(
            label="Download Matching Result",
            data=output.getvalue(),
            file_name="WIR_ITP_Matching.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.success("✅ Matching completed!")
