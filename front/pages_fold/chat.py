import os

import re
import streamlit as st
import logging as logging

import requests


# Initialize logging
logging.basicConfig(
    filename='app.log',           # Log file name
    level=logging.INFO,           # Log level
    format='%(asctime)s %(levelname)s:%(message)s'
)




def generate_response_server_revolvIA_tuto(input_message, user_id):

    url = "http://fastapi:8000/ask"
    payload = {
        "message": input_message,
        "user_id": user_id
    }

    response = requests.post(url, json=payload)
    print(response.status_code)
    back = response.json()
    print(back)

    return {"answer": back.get("answer"), "sources": back.get("sources")}


# Page layout
def page(user_id):

    st.title("💬 RevolvIA SOP Assistant")
    st.text("Talk with RevolvIA about SOP information!")

    st.markdown('<hr style="display: inline-block; width: 100%; margin-right: 10px;">', unsafe_allow_html=True)

    

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                for line in message["content"].splitlines():
                    match = re.match(r"\[IMAGE: (.+?)\]", line.strip())
                    if match:
                        st.image(match.group(1), use_column_width=True)
                    else:
                        st.markdown(line)
            else:
                st.markdown(message["content"])
    

    # Prompt input
    if prompt := st.chat_input("What is up?"):
        # Show user input
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get assistant response
        assistant_response = generate_response_server_revolvIA_tuto(prompt, user_id)
        

        answer = assistant_response["answer"] +"\n Sources: "+ str(assistant_response["sources"])

        source_match = re.search(
            r"source:\s*\[?([^\]\n]+\.(?:pdf|docx))\]?", answer, re.IGNORECASE
        )
        
        if source_match:
            source = source_match.group(1).strip()
            source_folder = os.path.splitext(source)[0]
        else:
            source_folder = "unknown_source"

        with st.chat_message("assistant"):
            def image_replacer(match):
                image_filename = match.group(1)
                image_path = f"extracted_images/{source_folder}/{image_filename}"

                logging.info(f"Source match: {image_path}")

                st.image(image_path, use_container_width=False)
        
                return ""

            pattern = r"\[(?:[^\[\]]*_)?(page_\d+_image_\d+\.(?:png|jpe?g))\]"

            for line in answer.splitlines():
                if re.search(pattern, line):
                    re.sub(pattern, image_replacer, line)
                else:
                    st.markdown(line)


        # Save assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response["answer"]
        })
