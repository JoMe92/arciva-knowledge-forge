
import streamlit as st
from managers.project import ProjectManager
from ui.home import render_home_screen
from ui.review import render_review_screen
from ui.sidebar import render_sidebar
from core.state import init_session_state

# Page Config
st.set_page_config(page_title="Arciva QA Review", layout="wide")

# Initialize State
init_session_state()

# Routing Logic
if st.session_state['active_project_id'] is None:
    # Project Management / Home
    render_home_screen()
else:
    # Active Project View
    pm = st.session_state['project_manager']
    project_id = st.session_state['active_project_id']
    project = pm.get_project(project_id)
    
    # Fallback/Recovery
    if not project:
        st.session_state['active_project_id'] = None
        st.rerun()

    # Render App
    render_sidebar(project)
    render_review_screen(project)
