import streamlit as st
import requests



def page(token):
    st.title("📂 Data Management - Document Uploader")

    uploaded_files = st.file_uploader(
        "Choose files (.csv, .txt, .pdf, .docx, .jpeg, .png)", 
        accept_multiple_files=True,
        type=["csv", "txt", "pdf", "docx", "doc", "jpg", "jpeg", "png"]
    )


    if uploaded_files and st.button("Upload to API"):
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]

        with st.status("Processing data..."):
            try:
                response = requests.post("http://api_agent:8000/upload_files/", files=files)
                if response.status_code == 200:
                    st.success("✅ Upload successful!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Failed with status code {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

