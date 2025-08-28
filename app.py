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
    "Upload 2 or more Excel files", type="xlsx", accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
    
    # Load all files
    dataframes = {}
    for file in uploaded_files:
        df = pd.read_excel(file)
        dataframes[file.name] = df
        st.subheader(f"Preview – {file.name}")
        st.dataframe(df.head())

    # -----------------------------
    # Step 2 – Select files & columns for matching
    # -----------------------------
    file1_name = st.selectbox("Select File 1", list(dataframes.keys()))
    file2_name = st.selectbox("Select File 2", list(dataframes.keys()))

    if file1_name != file2_name:
        df1, df2 = dataframes[file1_name], dataframes[file2_name]

        col1 = st.selectbox(f"Select column from {file1_name}", df1.columns)
        col2 = st.selectbox(f"Select column from {file2_name}", df2.columns)

        # -----------------------------
        # Step 3 – Perform Token-Based Matching
        # -----------------------------
        if st.button("Run Matching"):
            # Normalize function
            def normalize(text):
                if pd.isna(text):
                    return ""
                text = str(text).lower()
                text = re.sub(r'[^a-z0-9\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            df1['norm_col'] = df1[col1].apply(normalize)
            df2['norm_col'] = df2[col2].apply(normalize)
            df1['tokens'] = df1['norm_col'].str.split()

            results = []
            total_rows = len(df2)
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            for idx, row in enumerate(df2.itertuples(index=False), 1):
                norm_val = getattr(row, 'norm_col')
                if not norm_val:
                    continue
                tokens = set(norm_val.split())
                mask = df1['tokens'].apply(lambda x: tokens.issubset(set(x)))
                matched_rows = df1.loc[mask].copy()
                if not matched_rows.empty:
                    # Add all df2 columns
                    for i, col_name in enumerate(df2.columns):
                        matched_rows[col_name] = row[i]
                    results.append(matched_rows)

                # Update progress
                progress_bar.progress(idx / total_rows)
                status_text.text(f"Processing row {idx}/{total_rows} ({idx/total_rows*100:.1f}%)")
                time.sleep(0.01)

            end_time = time.time()
            final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

            if not final_df.empty:
                st.success(f"✅ Matching complete in {end_time - start_time:.2f} seconds")
                st.dataframe(final_df)

                # -----------------------------
                # Step 4 – Optional Filtering
                # -----------------------------
                st.subheader("Filter Matched Results (Optional)")
                filter_col = st.selectbox("Select column to filter", final_df.columns)
                keywords = st.text_input("Enter keywords (comma-separated)")

                if keywords:
                    terms = [k.strip().lower() for k in keywords.split(",")]
                    filtered_df = final_df[
                        final_df[filter_col].astype(str).str.lower().apply(
                            lambda x: any(term in x for term in terms)
                        )
                    ]
                    st.write(f"Filtered rows: {len(filtered_df)}")
                    st.dataframe(filtered_df)

                # -----------------------------
                # Step 5 – Download Results
                # -----------------------------
                def to_excel(df):
                    output = BytesIO()
                    df.to_excel(output, index=False)
                    return output.getvalue()

                st.download_button(
                    "💾 Download Full Matched Result",
                    data=to_excel(final_df),
                    file_name="matched_results.xlsx"
                )

                if keywords:
                    st.download_button(
                        "💾 Download Filtered Result",
                        data=to_excel(filtered_df),
                        file_name="filtered_results.xlsx"
                    )
            else:
                st.warning("⚠️ No matches found.")
