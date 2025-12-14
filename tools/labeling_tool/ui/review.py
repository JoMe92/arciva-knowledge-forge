
import streamlit as st
import json
import pandas as pd
from managers.data import DataManager
from ui.components import inject_shortcuts

def render_review_screen(project):
    # Paths from Project Config
    current_raw_path = project.get("raw_data_path")
    current_processed_path = project.get("processed_data_path")

    # View Switching
    mode = st.sidebar.radio("View Mode", ["Review", "Overview"], index=0)
    
    st.title("Arciva Dataset Review Tool")
    
    if mode == "Overview":
        render_overview(project, current_raw_path, current_processed_path)
    elif mode == "Review":
        render_review_session(project, current_raw_path, current_processed_path)

def render_overview(project, raw_path, processed_path):
    st.header("Project Model Card")
    st.markdown(f"**Project Name:** {project.get('name')}")
    st.markdown(f"**Description:** {project.get('description', 'No description.')}")
    st.markdown(f"**Raw Data:** `{raw_path}`")
    st.markdown(f"**Processed Data:** `{processed_path}`")
    
    st.markdown("---")
    st.header("Dataset Overview")
    
    df = DataManager.get_overview_data(raw_path, processed_path)
    
    if not df.empty:
        # Stats
        st.markdown("### Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pairs", len(df))
        c2.metric("Verified", len(df[df["Status"].str.contains("Verified")]))
        c3.metric("Modified", len(df[df["Status"].str.contains("Modified")]))
        c4.metric("Pending", len(df[df["Status"].str.contains("Pending")]))
        
        st.markdown("### Data Browser")
        
        # Filters
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            status_filter = st.multiselect(
                "Filter by Status", 
                options=["Verified", "Modified", "Discarded", "Pending"],
                default=[]
            )
        with f_col2:
            search_query = st.text_input("Search Question", placeholder="Type keywords...")
            
        # Apply Filters
        df_display = df.copy()
        if status_filter:
            pattern = "|".join(status_filter)
            df_display = df_display[df_display["Status"].str.contains(pattern, na=False)]
            
        if search_query:
            df_display = df_display[df_display["Question"].str.contains(search_query, case=False, na=False)]

        st.dataframe(
            df_display[["Question", "Status"]], 
            width="stretch",
            height=600,
            column_config={
                "Status": st.column_config.TextColumn("Status", help="Current review status")
            }
        )
    else:
        st.info("No data found.")

def render_review_session(project, raw_path, processed_path):
    inject_shortcuts()
    
    if len(st.session_state['data_queue']) > 0 and st.session_state['current_index'] < len(st.session_state['data_queue']):
        
        # Get current item
        item = st.session_state['data_queue'][st.session_state['current_index']]
        
        # Extract Q & A
        user_msg = next((m for m in item["messages"] if m["role"] == "user"), None)
        asst_msg = next((m for m in item["messages"] if m["role"] == "assistant"), None)
        
        if not user_msg or not asst_msg:
            st.error("Invalid message format in current item.")
            if st.button("Skip Invalid"):
                st.session_state['current_index'] += 1
                st.rerun()
        else:
            # Display User Question
            with st.chat_message("user"):
                st.write(user_msg["content"])
                
            # Display Assistant Answer (Editable)
            with st.chat_message("assistant"):
                new_answer = st.text_area(
                    "Edit Answer:", 
                    value=asst_msg["content"], 
                    height=300,
                    key=f"answer_{st.session_state['current_index']}", 
                    on_change=lambda: None 
                )
                
            # Review Controls
            st.markdown("### verification")
            
            if new_answer != asst_msg["content"]:
                current_status = "✍️ Modified"
                st.info("Status: **Modified** (Changes detected)")
            else:
                current_status = "✅ Verified"
                st.success("Status: **Verified** (No changes)")
                
            col1, col2 = st.columns([1, 1])
            
            save_format = project.get("save_format", "Multi-turn Dialog")
            
            # Discard Logic
            with col1:
                if st.button("🗑️ Discard Pair (Ctrl+Shift+Del)", type="secondary", width="stretch", help="Shortcut: Ctrl + Shift + Backspace/Delete"):
                     DataManager.save_entry(item, "🗑️ Discarded", new_answer, item, processed_path, save_format=save_format) 
                     st.session_state['current_index'] += 1
                     st.session_state['reviews_this_session'] += 1
                     st.rerun()
    
            # Save/Next Logic
            with col2:
                if st.button("Confirm & Next ➡️ (Shift+Enter)", type="primary", width="stretch", help="Shortcut: Shift + Enter"):
                    DataManager.save_entry(item, current_status, new_answer, item, processed_path, save_format=save_format)
                    st.session_state['current_index'] += 1
                    st.session_state['reviews_this_session'] += 1
                    st.rerun()
                
    elif len(st.session_state['data_queue']) > 0:
        st.success("Session Complete! 🎉")
        st.balloons()
        if st.button("Start New Session"):
            st.session_state['data_queue'] = []
            st.rerun()
    else:
        st.info("👈 Please load data from the sidebar.")
