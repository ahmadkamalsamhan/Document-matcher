# ------------------------------
# Streamlit WIR → ITP Tracker (Top 3 Matches)
# Memory-Efficient Version
# ------------------------------

import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import io

st.set_page_config(page_title="WIR → ITP Tracker", layout="wide")
st.title("WIR → ITP Activity Tracker (Top 3 Matches)")

# ------------------------------
# Upload Excel Files
# ------------------------------
st.header("Upload Excel Files")
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

        st.info("🔄 Expanding multi-activity WIR rows...")
        wir_df['ActivitiesList'] = wir_df['Title / Description2'].astype(str).str.split(r',|\+|/| and |&')
        wir_exp_df = wir_df.explode('ActivitiesList').reset_index(drop=True)
        wir_exp_df['ActivityNorm'] = wir_exp_df['ActivitiesList'].astype(str).str.upper().str.replace(" ", "")

        act_df['ActivityDescNorm'] = act_df['Activiy Description'].astype(str).str.upper().str.replace(" ", "")

        st.success(f"✅ Expanded WIR activities: {len(wir_exp_df)} rows total")

        st.info("🔍 Matching WIR activities to ITP activities...")

        progress_bar = st.progress(0)
        status_text = st.empty()
        matches = []

        # ------------------------------
        # Iterative Matching (Memory-Efficient)
        # ------------------------------
        total_rows = len(wir_exp_df)
        for idx, wir_row in wir_exp_df.iterrows():
            scores = []
            for _, itp_row in act_df.iterrows():
                score = fuzz.ratio(wir_row['ActivityNorm'], itp_row['ActivityDescNorm'])
                scores.append((itp_row['ITP Reference'], score, itp_row['Activiy Description']))

            # Sort top 3 matches
            scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

            # Keep matches above threshold
            top_matches = [(ref, desc, sc) if sc >= threshold else (None, desc, sc) for ref, sc, desc in scores_sorted]
            matches.append({
                'WIR Activity': wir_row['ActivitiesList'],
                'Top 1 ITP Reference': top_matches[0][0],
                'Top 1 Activity Description': top_matches[0][1],
                'Top 1 Score': top_matches[0][2],
                'Top 2 ITP Reference': top_matches[1][0],
                'Top 2 Activity Description': top_matches[1][1],
                'Top 2 Score': top_matches[1][2],
                'Top 3 ITP Reference': top_matches[2][0],
                'Top 3 Activity Description': top_matches[2][1],
                'Top 3 Score': top_matches[2][2],
            })

            if idx % 100 == 0:
                progress_bar.progress(idx / total_rows)
                status_text.text(f"Processed {idx}/{total_rows} WIR activities")

        progress_bar.progress(1.0)
        status_text.text("Matching completed!")

        match_df = pd.DataFrame(matches)
        st.subheader("✅ Top 3 Matches per WIR Activity")
        st.dataframe(match_df)

        # ------------------------------
        # Pivot Table
        # ------------------------------
        pivot_df = match_df.pivot_table(index='WIR Activity',
                                        values=['Top 1 Score', 'Top 2 Score', 'Top 3 Score'],
                                        aggfunc='max').reset_index()
        st.subheader("📊 Pivot Table of Top Scores")
        st.dataframe(pivot_df)

        # ------------------------------
        # Download Excel
        # ------------------------------
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            match_df.to_excel(writer, index=False, sheet_name='Top3Matches')
            pivot_df.to_excel(writer, index=False, sheet_name='PivotScores')
        output.seek(0)

        st.download_button(
            label="📥 Download Excel with Matches",
            data=output.getvalue(),
            file_name="WIR_ITP_Top3_Matches.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("Processing completed successfully!")
