
import streamlit as st
from datetime import datetime

DEFAULT_RAW_DATA_PATH = "data/raw/arciva_qa_synthetic.jsonl.dvc"
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
                        # Editable fields
                        new_name = st.text_input("Name", value=pdata['name'], key=f"rename_{pid}")
                        new_raw_path = st.text_input("Raw Data Path", value=pdata.get('raw_data_path', ''), key=f"repath_{pid}")
                        
                        st.markdown(f"**Description:** {pdata.get('description', '')}")
                        st.caption(f"Created: {datetime.fromtimestamp(pdata.get('created_at', 0)).strftime('%Y-%m-%d %H:%M')}")
                        
                        # Save Changes Button
                        if st.button("Save Changes", key=f"save_{pid}"):
                             updates = {}
                             if new_name != pdata['name']:
                                 updates["name"] = new_name
                             if new_raw_path != pdata.get('raw_data_path'):
                                 updates["raw_data_path"] = new_raw_path
                             
                             if updates:
                                 pm.update_project(pid, updates)
                                 st.success("Project updated!")
                                 st.rerun()

                    with c2:
                        if st.button("Open Project", key=f"open_{pid}", type="primary"):
                            st.session_state['active_project_id'] = pid
                            st.session_state['data_queue'] = [] 
                            st.rerun()
                        
                        st.divider()
                        
                        if st.button("🗑️ Delete Project", key=f"del_{pid}", type="secondary"):
                            pm.delete_project(pid)
                            st.success("Project deleted!")
                            st.rerun()

    with tab_create:
        st.subheader("Create New Project")
        c_name = st.text_input("Project Name")
        c_desc = st.text_area("Description")
        c_raw = st.text_input("Raw Data Path", value=DEFAULT_RAW_DATA_PATH)
        c_proc = st.text_input("Processed Data Path (Leave empty to edit Raw Data in-place)", value="")
        
        if st.button("Create Project"):
            if c_name and c_raw:
                # If processed path is not specified, default to writing back to the raw path (Edit in Place)
                final_proc = c_proc.strip() if c_proc.strip() else c_raw
                
                new_pid = pm.create_project(c_name, c_desc, c_raw, final_proc)
                st.session_state['active_project_id'] = new_pid
                st.success(f"Project '{c_name}' created!")
                st.session_state['data_queue'] = []
                st.rerun()
            else:
                st.error("Please fill in at least Project Name and Raw Data Path.")
