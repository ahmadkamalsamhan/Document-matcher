import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Load Sentence Transformer
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_text(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

# -------------------------------
# Match activity to WIR
# -------------------------------
def match_activity_to_wir(activity, candidate_wirs):
    best_match = None
    max_score = -1

    for idx, wir in candidate_wirs.iterrows():
        activity_tokens = set(preprocess_text(activity))
        wir_tokens = set(preprocess_text(wir['Title / Description2']))
        token_score = len(activity_tokens & wir_tokens)

        activity_emb = model.encode(activity, convert_to_tensor=True)
        wir_emb = model.encode(str(wir['Title / Description2']), convert_to_tensor=True)
        semantic_score = util.cos_sim(activity_emb, wir_emb).item()

        total_score = token_score + semantic_score

        if total_score > max_score:
            max_score = total_score
            best_match = wir

    return best_match

# -------------------------------
# Assign Status Code
# -------------------------------
def assign_status(wir):
    if wir is None or pd.isna(wir.get('PM Web Code')):
        return 0
    code = str(wir['PM Web Code']).strip().upper()
    if code in ['A','B']:
        return 1
    elif code in ['C','D']:
        return 2
    else:
        return 0

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 ITP-WIR Matching Tool")

st.write("Upload your Excel files for ITP Log, ITP Activities, and Document Control Log (WIR)")

# Upload files
itp_file = st.file_uploader("Upload ITP Log", type=["xlsx"])
activity_file = st.file_uploader("Upload ITP Activities Log", type=["xlsx"])
wir_file = st.file_uploader("Upload Document Control Log (WIR)", type=["xlsx"])

if itp_file and activity_file and wir_file:
    itp_log = pd.read_excel(itp_file)
    activity_log = pd.read_excel(activity_file)
    wir_log = pd.read_excel(wir_file)

    st.success("✅ Files uploaded successfully!")

    # Unique activities for columns
    unique_activities = activity_log['Activity Description'].dropna().unique().tolist()
    itp_nos = itp_log['DocumentNo.'].unique()
    matrix = pd.DataFrame(0, index=itp_nos, columns=unique_activities)

    progress = st.progress(0)
    total = len(itp_nos)
    
    for i, itp_no in enumerate(itp_nos):
        itp_title = itp_log.loc[itp_log['DocumentNo.']==itp_no, 'Title / Description'].values[0]
        activities = activity_log.loc[activity_log['ITP Reference']==itp_no]

        candidate_wirs = wir_log  # optionally filter by Function here

        for _, activity_row in activities.iterrows():
            activity_desc = activity_row['Activity Description']
            best_wir = match_activity_to_wir(activity_desc, candidate_wirs)
            status_code = assign_status(best_wir)
            matrix.at[itp_no, activity_desc] = status_code

        progress.progress((i+1)/total)

    matrix.reset_index(inplace=True)
    matrix.rename(columns={'index':'ITP No.'}, inplace=True)

    st.success("✅ ITP-WIR matrix generated successfully!")
    st.dataframe(matrix)

    # Download button
    st.download_button(
        label="📥 Download Matrix as Excel",
        data=matrix.to_excel(index=False, engine='openpyxl'),
        file_name="ITP_WIR_Matrix.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
