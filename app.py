# app.py
import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process, cdist
import io
import math

st.set_page_config(page_title="WIR → ITP Tracker", layout="wide")
st.title("📊 WIR ↔ ITP Activity Tracker")

# ------------------------------
# Upload Excel Files
# ------------------------------
st.header("Upload Excel Files")
wir_file = st.file_uploader("Upload WIR Log (.xlsx)", type=["xlsx"])
itp_file = st.file_uploader("Upload ITP Log (.xlsx)", type=["xlsx"])
activity_file = st.file_uploader("Upload ITP Activities Log (.xlsx)", type=["xlsx"])
threshold = st.slider("Fuzzy Match Threshold (%)", 70, 100, 90)

# ------------------------------
# Split WIR into 4 parts
# ------------------------------
if wir_file:
    wir_df = pd.read_excel(wir_file)
    st.success(f"Uploaded WIR: {len(wir_df)} rows")
    
    num_parts = 4
    part_size = math.ceil(len(wir_df) / num_parts)
    wir_parts = []
    
    st.subheader("Download WIR Parts")
    for i in range(num_parts):
        start = i * part_size
        end = min((i+1) * part_size, len(wir_df))
        part_df = wir_df.iloc[start:end].reset_index(drop=True)
        wir_parts.append(part_df)

        # Prepare Excel for download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            part_df.to_excel(writer, index=False, sheet_name=f"WIR_part{i+1}")
        output.seek(0)

        st.download_button(
            label=f"Download WIR Part {i+1}",
            data=output.getvalue(),
            file_name=f"WIR_part{i+1}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ------------------------------
# Start Processing Button
# ------------------------------
if wir_file and itp_file and activity_file:
    if st.button("Start Processing All Parts"):
        st.info("Reading Excel files...")
        itp_df = pd.read_excel(itp_file)
        act_df = pd.read_excel(activity_file)

        # Clean columns
        def clean_columns(df):
            df.columns = df.columns.str.strip()
            df.columns = df.columns.str.replace('\n',' ').str.replace('\r',' ')
            return df

        wir_df = clean_columns(wir_df)
        itp_df = clean_columns(itp_df)
        act_df = clean_columns(act_df)

        # Detect columns
        wir_pm_col = [c for c in wir_df.columns if 'PM Web Code' in c][0]
        wir_title_col = [c for c in wir_df.columns if 'Title / Description2' in c][0]

        act_itp_col = [c for c in act_df.columns if 'ITP Reference' in c][0]
        act_desc_col = [c for c in act_df.columns if 'Activiy Description' in c][0]

        # Normalize ITP Activities
        act_df['ITP_Ref_Norm'] = act_df[act_itp_col].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")
        act_df['ActivityDescNorm'] = act_df[act_desc_col].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")

        final_matches = []

        # Process each WIR part
        for idx, part_df in enumerate(wir_parts):
            st.info(f"🔄 Processing WIR Part {idx+1}")
            part_df['PM Web Code'] = part_df[wir_pm_col]
            part_df['PM_Code_Num'] = part_df['PM Web Code'].map(lambda x: 1 if str(x).upper() in ['A','B'] else 2 if str(x).upper() in ['C','D'] else 0)
            
            # Expand multi-activity WIR titles
            part_df['ActivitiesList'] = part_df[wir_title_col].astype(str).str.split(r',|\+|/| and |&')
            wir_exp_df = part_df.explode('ActivitiesList').reset_index(drop=True)
            wir_exp_df['ActivityNorm'] = wir_exp_df['ActivitiesList'].astype(str).str.upper().str.strip().str.replace("-", "").str.replace(" ", "")
            st.success(f"Expanded WIR activities: {len(wir_exp_df)} rows")

            # Vectorized Matching
            st.info(f"🔍 Matching WIR Part {idx+1} to ITP activities...")
            wir_list = wir_exp_df['ActivityNorm'].tolist()
            wir_pm_list = wir_exp_df['PM_Code_Num'].tolist()
            itp_list = act_df['ActivityDescNorm'].tolist()

            sim_matrix = cdist(itp_list, wir_list, scorer=fuzz.ratio)
            best_idx = np.argmax(sim_matrix, axis=1)
            best_score = np.max(sim_matrix, axis=1)
            pm_codes = [wir_pm_list[i] if s >= threshold else 0 for i,s in zip(best_idx, best_score)]

            match_df = pd.DataFrame({
                'ITP Reference': act_df[act_itp_col],
                'Activity Description': act_df[act_desc_col],
                'Status': pm_codes
            })

            audit_df = pd.DataFrame({
                'ITP Reference': act_df[act_itp_col],
                'Activity Description': act_df[act_desc_col],
                'Best Match': [wir_exp_df['ActivityNorm'][i] for i in best_idx],
                'Score': best_score
            })
            audit_df = audit_df[audit_df['Score'] < threshold]

            final_matches.append((match_df, audit_df))

        # Combine results
        all_match_df = pd.concat([m[0] for m in final_matches]).groupby(['ITP Reference', 'Activity Description']).max().reset_index()
        all_audit_df = pd.concat([m[1] for m in final_matches]).reset_index(drop=True)

        # Pivot Table
        pivot_df = all_match_df.pivot_table(index='ITP Reference',
                                            columns='Activity Description',
                                            values='Status',
                                            fill_value=0).reset_index()
        st.subheader("Activity Completion Status Pivot Table")
        st.dataframe(pivot_df)

        # Excel Output
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, index=False, sheet_name='PivotedStatus')
            if not all_audit_df.empty:
                all_audit_df.to_excel(writer, index=False, sheet_name='Audit_LowConfidence')
        output.seek(0)

        st.download_button(
            label="Download Full Result Excel",
            data=output.getvalue(),
            file_name="ITP_WIR_Activity_Status.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("✅ Processing completed successfully!")
