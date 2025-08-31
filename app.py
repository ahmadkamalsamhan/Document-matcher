# ------------------------------
# Ultra High-Performance WIR → ITP Tracker (Batch Matching)
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, cdist
import io

st.set_page_config(page_title="Ultra High-Performance WIR → ITP Tracker", layout="wide")
st.title("📊 WIR → ITP Activity Tracker")

# ------------------------------
# Upload Excel Files
# ------------------------------
st.header("Upload Excel Files")
wir_file = st.file_uploader("Upload WIR Log (.xlsx)", type=["xlsx"])
itp_file = st.file_uploader("Upload ITP Log (.xlsx)", type=["xlsx"])
activity_file = st.file_uploader("Upload ITP Activities Log (.xlsx)", type=["xlsx"])

threshold = st.slider("Fuzzy Match Threshold (%)", 70, 100, 90)

# ------------------------------
# Start Processing Button
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
            df.columns = df.columns.str.strip()
            df.columns = df.columns.str.replace('\n',' ').str.replace('\r',' ')
            return df

        wir_df = clean_columns(wir_df)
        itp_df = clean_columns(itp_df)
        act_df = clean_columns(act_df)

        # ------------------------------
        # Detect Columns
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
        # Batch Matching using RapidFuzz
        # ------------------------------
        st.info("🔍 Matching WIR activities to ITP activities...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        batch_size = 1000
        num_batches = int(np.ceil(len(wir_exp_df)/batch_size))

        best_matches = []
        best_scores = []

        itp_list = act_df['ActivityDescNorm'].tolist()
        wir_pm_list = wir_exp_df['PM_Code_Num'].tolist()

        for i in range(num_batches):
            start = i*batch_size
            end = min((i+1)*batch_size, len(wir_exp_df))
            batch_wir = wir_exp_df['ActivityNorm'][start:end].tolist()

            # Compute similarity for this batch vs all ITP activities
            sim_matrix = cdist(batch_wir, itp_list, scorer=fuzz.ratio)

            for j in range(sim_matrix.shape[0]):
                top_idx = np.argmax(sim_matrix[j])
                best_matches.append(top_idx)
                best_scores.append(sim_matrix[j, top_idx])

            progress_bar.progress((i+1)/num_batches)
            status_text.text(f"Processed {end}/{len(wir_exp_df)} WIR activities...")

        # ------------------------------
        # Build matches DataFrame
        # ------------------------------
        pm_codes = [wir_pm_list[i] if s >= threshold else 0 for i,s in zip(best_matches, best_scores)]
        match_df = pd.DataFrame({
            'ITP Reference': act_df[act_itp_col].iloc[best_matches].values,
            'Activity Description': act_df[act_desc_col].iloc[best_matches].values,
            'Status': pm_codes
        })

        # Audit sheet for low-confidence matches
        audit_df = pd.DataFrame({
            'ITP Reference': act_df[act_itp_col].iloc[best_matches].values,
            'Activity Description': act_df[act_desc_col].iloc[best_matches].values,
            'Best Match': wir_exp_df['ActivityNorm'].values,
            'Score': best_scores
        })
        audit_df = audit_df[audit_df['Score'] < threshold]

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
            label="📥 Download Excel",
            data=output.getvalue(),
            file_name="ITP_WIR_Activity_Status.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("✅ Processing completed successfully!")
