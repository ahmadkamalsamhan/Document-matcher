# ------------------------------
# Ultra High-Performance WIR ↔ ITP Tracker
# Vectorized Matching with RapidFuzz cdist
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process, cdist
import io
import time

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Ultra High-Performance WIR → ITP Tracker", layout="wide")
st.title("📊 WIR → ITP Activity Tracking Tool (Ultra High Performance)")

# ------------------------------
# File Upload
# ------------------------------
st.header("📁 Upload Excel Files")
wir_file = st.file_uploader("Upload WIR Log (.xlsx)", type=["xlsx"])
itp_file = st.file_uploader("Upload ITP Log (.xlsx)", type=["xlsx"])
activity_file = st.file_uploader("Upload ITP Activities Log (.xlsx)", type=["xlsx"])

threshold = st.slider("Fuzzy Match Threshold (%)", 70, 100, 90)

# ------------------------------
# Start Processing
# ------------------------------
if wir_file and itp_file and activity_file:
    if st.button("Start Processing"):

        st.info("📖 Reading Excel files...")
        wir_df = pd.read_excel(wir_file)
        itp_df = pd.read_excel(itp_file)
        act_df = pd.read_excel(activity_file)

        # ------------------------------
        # Clean Columns
        # ------------------------------
        def clean_columns(df):
            df.columns = df.columns.str.strip().str.replace('\n',' ').str.replace('\r',' ')
            return df

        wir_df = clean_columns(wir_df)
        itp_df = clean_columns(itp_df)
        act_df = clean_columns(act_df)

        # ------------------------------
        # Detect Relevant Columns
        # ------------------------------
        wir_pm_col = [c for c in wir_df.columns if 'PM Web Code' in c][0]
        wir_title_col = [c for c in wir_df.columns if 'Title / Description2' in c][0]

        act_itp_col = [c for c in act_df.columns if 'ITP Reference' in c][0]
        act_desc_col = [c for c in act_df.columns if 'Activiy Description' in c][0]

        # ------------------------------
        # Normalize WIR
        # ------------------------------
        wir_df['PM Web Code'] = wir_df[wir_pm_col]
        wir_df['PM_Code_Num'] = wir_df['PM Web Code'].map(lambda x: 1 if str(x).upper() in ['A','B'] else 2 if str(x).upper() in ['C','D'] else 0)

        st.info("🔄 Expanding multi-activity WIR rows...")
        wir_df['ActivitiesList'] = wir_df[wir_title_col].astype(str).str.split(r',|\+|/| and |&')
        wir_exp_df = wir_df.explode('ActivitiesList').reset_index(drop=True)
        wir_exp_df['ActivityNorm'] = wir_exp_df['ActivitiesList'].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")
        st.success(f"✅ Expanded WIR activities: {len(wir_exp_df)} rows total")

        # ------------------------------
        # Normalize ITP Activities
        # ------------------------------
        act_df['ITP_Ref_Norm'] = act_df[act_itp_col].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")
        act_df['ActivityDescNorm'] = act_df[act_desc_col].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")

        # ------------------------------
        # Vectorized Matching using RapidFuzz cdist
        # ------------------------------
        st.info("🔍 Matching WIR activities to ITP activities...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        wir_list = wir_exp_df['ActivityNorm'].tolist()
        wir_pm_list = wir_exp_df['PM_Code_Num'].tolist()
        itp_list = act_df['ActivityDescNorm'].tolist()

        # Batch processing to show progress
        batch_size = 1000
        num_batches = int(np.ceil(len(wir_list)/batch_size))

        best_idx_list = []
        best_score_list = []

        for i in range(num_batches):
            start = i*batch_size
            end = min((i+1)*batch_size, len(wir_list))
            batch = wir_list[start:end]
            sim_matrix = cdist(itp_list, batch, scorer=fuzz.ratio)
            best_idx = np.argmax(sim_matrix, axis=0)
            best_score = np.max(sim_matrix, axis=0)
            best_idx_list.extend(best_idx.tolist())
            best_score_list.extend(best_score.tolist())
            progress_bar.progress((i+1)/num_batches)
            status_text.text(f"Processed {end}/{len(wir_list)} WIR activities...")
            time.sleep(0.01)  # small sleep to allow Streamlit UI update

        # Determine PM codes based on threshold
        pm_codes = [wir_pm_list[i] if s >= threshold else 0 for i,s in zip(range(len(best_score_list)), best_score_list)]

        # ------------------------------
        # Build Matches DataFrame
        # ------------------------------
        match_df = pd.DataFrame({
            'ITP Reference': act_df[act_itp_col],
            'Activity Description': act_df[act_desc_col],
            'Status': pm_codes
        })

        # ------------------------------
        # Build Audit for Low Confidence Matches
        # ------------------------------
        audit_df = pd.DataFrame({
            'ITP Reference': act_df[act_itp_col],
            'Activity Description': act_df[act_desc_col],
            'Best Match': [wir_exp_df['ActivityNorm'][i] for i in best_idx_list],
            'Score': best_score_list
        })
        audit_df = audit_df[audit_df['Score'] < threshold]

        progress_bar.progress(1.0)
        status_text.text("✅ Matching completed!")

        # ------------------------------
        # Pivot Table
        # ------------------------------
        pivot_df = match_df.pivot_table(index='ITP Reference',
                                        columns='Activity Description',
                                        values='Status',
                                        fill_value=0).reset_index()

        st.subheader("Activity Completion Status Pivot Table")
        st.dataframe(pivot_df)

        # ------------------------------
        # Excel Output with Audit Sheet
        # ------------------------------
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, index=False, sheet_name='PivotedStatus')
            if not audit_df.empty:
                audit_df.to_excel(writer, index=False, sheet_name='Audit_LowConfidence')
        output.seek(0)

        st.download_button(
            label="💾 Download Excel",
            data=output.getvalue(),
            file_name="ITP_WIR_Activity_Status.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("🎉 Processing completed successfully!")
