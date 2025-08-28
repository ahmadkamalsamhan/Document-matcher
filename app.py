# -----------------------------
# Reset / Clear Uploaded Files safely
# -----------------------------
reset_clicked = st.button("🗑 Clear Uploaded Files / Reset App")
if reset_clicked:
    keys_to_clear = ["uploaded_files", "df1_small", "df2_small", "tmp_path"]
    cleared = False
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
            cleared = True
    if cleared:
        st.success("✅ Uploaded files and session cleared. You can start fresh.")
        st.experimental_rerun()  # rerun only if we actually cleared something
    else:
        st.success("✅ App is already clean. You can continue normally.")
