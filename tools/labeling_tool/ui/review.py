
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
        # --- Metrics & Progress ---
        total = len(df)
        verified = len(df[df["Status"].str.contains("Verified", na=False)])
        modified = len(df[df["Status"].str.contains("Modified", na=False)])
        discarded = len(df[df["Status"].str.contains("Discarded", na=False)])
        pending = len(df[df["Status"].str.contains("Pending", na=False)])
        
        completed = verified + modified + discarded
        pct_complete = (completed / total) if total > 0 else 0
        
        st.markdown(f"### Overall Progress: {int(pct_complete*100)}%")
        st.progress(pct_complete)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Items", total)
        m2.metric("✅ Verified", verified)
        m3.metric("✍️ Modified", modified)
        m4.metric("🗑️ Discarded", discarded)
        m5.metric("⏳ Pending", pending)
        
        st.markdown("---")
        
        # --- Visualizations ---
        st.subheader("📈 Analytics")
        
        # 1. Label Distribution
        st.markdown("**Label Distribution**")
        status_counts = df["Status"].value_counts()
        st.bar_chart(status_counts)
        
        # 2. Length Analysis
        st.markdown("**Message Length Analysis**")
        
        # Helper to safely get lengths
        def get_lengths(row):
            # Try getting from Reviewed Entry first
            entry = row.get("Reviewed_Entry")
            if not isinstance(entry, dict) or not entry:
                entry = row.get("Raw_Entry")
            
            u_len = 0
            a_len = 0
            if isinstance(entry, dict):
                msgs = entry.get("messages", [])
                u_content = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                a_content = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
                u_len = len(u_content) if u_content else 0
                a_len = len(a_content) if a_content else 0
            return pd.Series([u_len, a_len])

        length_df = df.apply(get_lengths, axis=1)
        length_df.columns = ["User Input", "Assistant Output"]
        st.line_chart(length_df)
        
        # 3. Time Series (if available)
        # Extract timestamps from reviewed entries
        timestamps = []
        for _, row in df.iterrows():
            entry = row.get("Reviewed_Entry")
            if isinstance(entry, dict):
                meta = entry.get("review_metadata", {})
                ts = meta.get("timestamp")
                if ts:
                    timestamps.append(ts)
        
        if timestamps:
            st.markdown("**Review Activity**")
            ts_df = pd.DataFrame({"Timestamp": pd.to_datetime(timestamps, unit='s')})
            ts_df["Count"] = 1
            ts_df.set_index("Timestamp", inplace=True)
            # Resample to Hour or Minute depending on density
            activity = ts_df.resample('h').sum()
            st.bar_chart(activity)

        st.markdown("---")
        st.header("Data Browser")
        
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
