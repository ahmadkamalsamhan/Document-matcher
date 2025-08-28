import streamlit as st
import pandas as pd
import re
from io import BytesIO
import time

st.set_page_config(page_title="King Salman Park Matching App", layout="wide")
st.title("📊 King Salman Park - Token-Based Matching App")

# -----------------------------
# Step 1 – Upload Files
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Excel files", type="xlsx", accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

    # Show checkboxes to select files
    st.subheader("Select files to use for matching")
    selected_files = []
    for file in uploaded_files:
        if st.checkbox(file.name, value=True):
            selected_files.append(file)

    if len(selected_files) >= 2:
        # Load first two selected files
        df1 = pd.read_excel(selected_files[0])
        df2 = pd.read_excel(selected_files[1])

        st.write(f"Preview – {selected_files[0].name}")
        st.dataframe(df1.head())
        st.write(f"Preview – {selected_files[1].name}")
        st.dataframe(df2.head())

        # -----------------------------
        # Step 2 – Column Selection
        # -----------------------------
        st.subheader("Select Columns to Match")
        cols1 = st.multiselect(f"Select column(s) from {selected_files[0].name}", df1.columns)
        cols2 = st.multiselect(f"Select column(s) from {selected_files[1].name}", df2.columns)

        if cols1 and cols2:
            if st.button("Start Matching"):
                st.info("⏳ Matching in progress...")

                # Normalize function
                def normalize(text):
                    if pd.isna(text):
                        return ""
                    text = str(text).lower()
                    text = re.sub(r'[^a-z0-9\s]', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text

                # Normalize columns in df1
                for col in cols1:
                    df1[f"norm_{col}"] = df1[col].apply(normalize)
                    df1[f"tokens_{col}"] = df1[f"norm_{col}"].str.split()

                # Normalize columns in df2
                for col in cols2:
                    df2[f"norm_{col}"] = df2[col].apply(normalize)

                results = []
                total_rows = len(df2)
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                # Iterate rows in df2
                for idx, row in df2.iterrows():
                    row_matches = pd.DataFrame()
                    for col1 in cols1:
                        for col2 in cols2:
                            tokens = set(row[f"norm_{col2}"].split())
                            if not tokens:
                                continue
                            mask = df1[f"tokens_{col1}"].apply(lambda x: tokens.issubset(set(x)))
                            matched_rows = df1.loc[mask].copy()
                            if not matched_rows.empty:
                                for c in df2.columns:
                                    matched_rows[c] = row[c]
                                row_matches = pd.concat([row_matches, matched_rows], ignore_index=True)

                    if not row_matches.empty:
                        results.append(row_matches)

                    progress_bar.progress((idx + 1) / total_rows)
                    status_text.text(f"Processing row {idx+1}/{total_rows} ({(idx+1)/total_rows*100:.1f}%)")
                    time.sleep(0.01)

                end_time = time.time()
                final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

                if not final_df.empty:
                    st.success(f"✅ Matching complete in {end_time - start_time:.2f} seconds")
                    st.dataframe(final_df)

                    # Download button
                    output = BytesIO()
                    final_df.to_excel(output, index=False)
                    st.download_button(
                        "💾 Download Matched Results",
                        data=output.getvalue(),
                        file_name="matched_results.xlsx"
                    )
                else:
                    st.warning("⚠️ No matches found.")
        else:
            st.warning("⚠️ Please select at least one column from each file.")
    else:
        st.warning("⚠️ Please select at least 2 files for matching.")
