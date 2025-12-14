
import streamlit as st
from datetime import datetime

DEFAULT_RAW_DATA_PATH = "data/raw/arciva_qa_synthetic.jsonl"
DEFAULT_PROCESSED_DATA_PATH = "data/processed/arciva_qa_reviewed.jsonl"

def render_home_screen():
    st.title("🗂️ Project Management")
    
    pm = st.session_state['project_manager']
    projects = pm.load_projects()

    tab_list, tab_create = st.tabs(["Open Project", "Create New Project"])
    
    with tab_list:
        if not projects:
            st.info("No projects found. Create one to get started!")
        else:
            for pid, pdata in projects.items():
                with st.expander(f"📁 {pdata['name']}", expanded=False):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Description:** {pdata.get('description', '')}")
                        st.caption(f"Created: {datetime.fromtimestamp(pdata.get('created_at', 0)).strftime('%Y-%m-%d %H:%M')}")
                        st.caption(f"Raw Data: `{pdata.get('raw_data_path')}`")
                    with c2:
                        if st.button("Open Project", key=f"open_{pid}"):
                            st.session_state['active_project_id'] = pid
                            st.session_state['data_queue'] = [] 
                            st.rerun()
                        
                        new_name = st.text_input("Rename:", value=pdata['name'], key=f"rename_{pid}")
                        if new_name != pdata['name']:
                             if st.button("Save Name", key=f"save_name_{pid}"):
                                 pm.update_project(pid, {"name": new_name})
                                 st.success("Renamed!")
                                 st.rerun()

    with tab_create:
        st.subheader("Create New Project")
        c_name = st.text_input("Project Name")
        c_desc = st.text_area("Description")
        c_raw = st.text_input("Raw Data Path", value=DEFAULT_RAW_DATA_PATH)
        c_proc = st.text_input("Processed Data Path", value=DEFAULT_PROCESSED_DATA_PATH)
        
        if st.button("Create Project"):
            if c_name and c_raw and c_proc:
                new_pid = pm.create_project(c_name, c_desc, c_raw, c_proc)
                st.session_state['active_project_id'] = new_pid
                st.success(f"Project '{c_name}' created!")
                st.session_state['data_queue'] = []
                st.rerun()
            else:
                st.error("Please fill in all required fields.")
