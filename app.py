import streamlit as st
import pandas as pd
import io
import math

st.set_page_config(page_title="WIR Splitter & Tracker", layout="wide")
st.title("🔹 WIR Splitter & Tracker")

# ------------------------------
# Upload Full WIR Excel
# ------------------------------
wir_file = st.file_uploader("Upload Full WIR Log (.xlsx)", type=["xlsx"])

if wir_file:
    st.info("Reading WIR Excel file...")
    wir_df = pd.read_excel(wir_file)
    
    # Clean columns
    wir_df.columns = wir_df.columns.str.strip().str.replace('\n',' ').str.replace('\r',' ')
    
    st.success(f"✅ WIR file loaded with {len(wir_df)} rows")

    # ------------------------------
    # Split WIR into 4 parts
    # ------------------------------
    num_parts = 4
    part_size = math.ceil(len(wir_df) / num_parts)
    
    parts = []
    for i in range(num_parts):
        start = i * part_size
        end = min((i+1) * part_size, len(wir_df))
        part_df = wir_df.iloc[start:end].reset_index(drop=True)
        parts.append(part_df)
    
    st.subheader("Download WIR Parts")
    for idx, part_df in enumerate(parts):
        output = io.BytesIO()
        part_df.to_excel(output, index=False, sheet_name=f'WIR_Part{idx+1}')
        output.seek(0)
        st.download_button(
            label=f"Download WIR Part {idx+1}",
            data=output.getvalue(),
            file_name=f"WIR_part{idx+1}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    st.info("✅ WIR file split into 4 parts. Download and process each part individually.")
