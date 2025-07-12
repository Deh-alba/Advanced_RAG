import streamlit as st
import pages_fold.chat as chat
import pages_fold.data_management as data_management
import getpass

from pam import pam
import pexpect
import time





def sleep(seconds):
    import time
    time.sleep(seconds)



def update_page_bool(flag):
    if flag == "chat":
        st.session_state.chat = True
        st.session_state.data_management = False



    elif flag == "data_management":
        st.session_state.chat = False
        st.session_state.data_management = True

    


@st.dialog("Renew Password", width="large")
def reset_password():
    username = st.session_state.get("user_id", getpass.getuser())
    
    old_password = st.text_input("Old Password", type="password")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm New Password", type="password")

    if st.button("Save New Password"):
        if new_password != confirm_password:
            st.error("Passwords do not match. Please try again.")
            return

        try:
            logfile_path = "/tmp/pexpect_passwd.log"
            with open(logfile_path, "w") as log_file:
                child = pexpect.spawn(f'passwd {username}', timeout=10)
                child.expect(".*[Cc]urrent.*password.*:")
                child.sendline(old_password)

                child.expect(".*[Nn]ew.*password.*:")
                child.sendline(new_password)

                child.expect(".*[Rr]etype.*new.*password.*:")
                child.sendline(new_password)

                child.expect(pexpect.EOF)
                output = child.before.decode()

            if "success" in output.lower() or "updated successfully" in output.lower():
                st.success("Password renewed successfully!")
                time.sleep(3)
                st.session_state.pop("auth_token", None)
                st.success("Logged out!")
                st.rerun()
            else:
                st.error("Password change failed. Check your credentials or try again.")
                st.text(output)  # Optionally show command output for debugging

        except pexpect.exceptions.TIMEOUT:
            st.error(f"Timeout during password change. Please try again. {logfile_path}")
        except pexpect.exceptions.EOF:
            st.error(f"❌ Erro inesperado: check your credentials or try again. {logfile_path}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")



def login_element():

    login_element = st.empty()
    column1, column2, column3 = login_element.columns(3)
    login_container = column2.container(border=True)


    login_container.title("Login")

    username = login_container.text_input("Username")
    password = login_container.text_input("Password", type="password")
    username = username.strip()

    auth = pam()

    if login_container.button("Login"):

        
        if auth.authenticate(username, password):
            
            response = {'status_code':200}

            st.session_state["auth_token"] = 1234
            st.session_state["user_id"] = username

            login_container.success("Logged in!")
            st.rerun()
        else:
            response = {'status_code':401}
            login_container.error("An unknown error occurred. Please try again later.")


#from utils import logo

#logo()
# Hide the default Streamlit menu
st.set_page_config(
    page_title="Agent", 
    page_icon=":material/edit:", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
    )



if "chat" not in st.session_state:
    st.session_state.chat=False

if "data_management" not in st.session_state:
    st.session_state.data_management=False


if "auth_token" not in st.session_state:
    login_element()

if "auth_token" in st.session_state:  


    #token_valid = token_validation(st.session_state["auth_token"])
    token_valid = True  # Simulating a valid token for demonstration purposes
    
    if token_valid:

        with st.sidebar:

            
            st.markdown('<hr style="display: inline-block; width: 100%; margin-right: 10px;">', unsafe_allow_html=True)

            
            telas = st.container(border=True)

            if telas.button("## Agent", type='tertiary',icon="📈", use_container_width=True):
                update_page_bool("chat")

            telas.markdown('<hr style="display: inline-block; width: 100%; margin-right: 10px;">', unsafe_allow_html=True)

            if telas.button("## Data Upload", type='tertiary',icon="🖐️", use_container_width=True):
                update_page_bool("data_management")
            

            buttons_login_passwd = st.container(border=True)
            col1, col2 = buttons_login_passwd.columns(2)


            if col1.button("Leave", icon="🚪", use_container_width=True):
                st.session_state.pop("auth_token", None)
                st.success("Logged out!")
                st.rerun()
                            


        if st.session_state.chat and token_valid:
            chat.page(st.session_state["user_id"])
        elif st.session_state.data_management and token_valid:
            data_management.page(st.session_state["user_id"])

    
    else:

        st.error("Token invalid or expired")
        sleep(5)
        st.rerun()


