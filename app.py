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
        # -----------------------------
        # Step 2 – Load & preview first two selected files
        # -----------------------------
        df1 = pd.read_excel(selected_files[0])
        df2 = pd.read_excel(selected_files[1])

        st.write(f"Preview – {selected_files[0].name}")
        st.dataframe(df1.head())
        st.write(f"Preview – {selected_files[1].name}")
        st.dataframe(df2.head())

        # Fixed columns like Colab version
        dc_columns = ["DocumentNo.", "Title / Description", "Function", "Discipline",
                      "Document Revision", "(Final Status)\nOpen or Closed"]
        details_columns = ["Package", "Phase", "Part", "Managers", "Service", "Line",
                           "Description", "Start Structure", "End Structure", "MH Type", "Date"]

        df1 = df1[dc_columns]
        df2 = df2[details_columns]

        # -----------------------------
        # Step 3 – Start Matching Button
        # -----------------------------
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

            df1['norm_title'] = df1["Title / Description"].apply(normalize)
            df1['title_tokens'] = df1['norm_title'].str.split()
            df2['norm_mh'] = df2["Start Structure"].apply(normalize)

            results = []
            total_rows = len(df2)
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            # Use iterrows() to avoid AttributeError
            for idx, row in df2.iterrows():
                mh_tokens = set(row['norm_mh'].split())
                if not mh_tokens:
                    continue
                mask = df1['title_tokens'].apply(lambda x: mh_tokens.issubset(set(x)))
                matched_rows = df1.loc[mask].copy()
                if not matched_rows.empty:
                    for col in details_columns:
                        matched_rows[col] = row[col]
                    results.append(matched_rows)

                progress_bar.progress((idx + 1) / total_rows)
                status_text.text(f"Processing row {idx+1}/{total_rows} ({(idx+1)/total_rows*100:.1f}%)")
                time.sleep(0.01)

            end_time = time.time()
            final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

            if not final_df.empty:
                final_df = final_df[dc_columns + details_columns]
                final_df = final_df.rename(columns={"(Final Status)\nOpen or Closed": "Final Status"})
                st.success(f"✅ Matching complete in {end_time - start_time:.2f} seconds")
                st.dataframe(final_df)

                # -----------------------------
                # Step 4 – Download
                # -----------------------------
                output = BytesIO()
                final_df.to_excel(output, index=False)
                st.download_button(
                    "💾 Download Matched Results",
                    data=output.getvalue(),
                    file_name="matched_non_merged_fast.xlsx"
                )
            else:
                st.warning("⚠️ No matches found.")
    else:
        st.warning("⚠️ Please select at least 2 files for matching.")
