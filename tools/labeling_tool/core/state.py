
import streamlit as st
import time
from managers.project import ProjectManager

def init_session_state():
    if 'project_manager' not in st.session_state:
        st.session_state['project_manager'] = ProjectManager()
    if 'active_project_id' not in st.session_state:
        st.session_state['active_project_id'] = None

    if 'data_queue' not in st.session_state:
        st.session_state['data_queue'] = []
    if 'current_index' not in st.session_state:
        st.session_state['current_index'] = 0
    if 'reviews_this_session' not in st.session_state:
        st.session_state['reviews_this_session'] = 0
    if 'session_start_time' not in st.session_state:
        st.session_state['session_start_time'] = time.time()
