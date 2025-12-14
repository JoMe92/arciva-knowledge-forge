
import streamlit as st
import time
import os
import json
import pandas as pd
from managers.data import DataManager

def render_sidebar(project):
    with st.sidebar:
        if st.button("🔙 Back to Projects"):
            st.session_state['active_project_id'] = None
            st.rerun()
        
        st.title(f"📂 {project['name']}")
        st.caption("Active Project")
    
        st.markdown("---")
        st.title("⚙️ Config")
        
        # Save Format
        current_format = project.get("save_format", "Multi-turn Dialog")
        new_format = st.selectbox("Save Format", ["Multi-turn Dialog", "Alpaca"], index=0 if current_format == "Multi-turn Dialog" else 1)
        
        if new_format != current_format:
            st.session_state['project_manager'].update_project(st.session_state['active_project_id'], {"save_format": new_format})
            st.rerun()

        # Paths from Project
        current_raw_path = project.get("raw_data_path")
        current_processed_path = project.get("processed_data_path")

        # Sampling Controls
        # Only show if queue is empty or reset requested
        if len(st.session_state['data_queue']) == 0:
            sample_pct = st.slider("Sample Size (%)", 1, 100, 10)
            include_reviewed = st.checkbox("Include previously labeled data", help="Include already verified items in the session (will be skipped/marked as done).")
            
            if st.button("Load Session"):
                queue, total_avail, total_reviewed = DataManager.load_data(current_raw_path, current_processed_path, sample_pct/100.0, include_reviewed=include_reviewed)
                if not queue and total_avail == 0:
                    st.error(f"No data found at {current_raw_path}")
                else:
                    st.session_state['data_queue'] = queue
                    st.session_state['total_avail_count'] = total_avail
                    st.session_state['total_reviewed_history'] = total_reviewed
                    
                    if include_reviewed:
                         st.session_state['current_index'] = min(total_reviewed, len(queue))
                    else:
                         st.session_state['current_index'] = 0

                    st.session_state['reviews_this_session'] = 0
                    st.rerun()
                
        else:
            if st.button("Reset / New Session"):
                st.session_state['data_queue'] = []
                st.rerun()

        st.markdown("---")
        st.title("📊 Dashboard")
        
        if len(st.session_state['data_queue']) > 0:
            queue_len = len(st.session_state['data_queue'])
            current = st.session_state['current_index']
            
            # Progress
            progress = current / queue_len
            st.progress(progress)
            st.write(f"Progress: {current} / {queue_len} ({int(progress*100)}%)")
            
            # Stats
            elapsed = time.time() - st.session_state['session_start_time']
            avg_time = elapsed / max(1, st.session_state['reviews_this_session'])
            remaining = (queue_len - current) * avg_time
            
            st.metric("Avg Time / Pair", f"{avg_time:.1f}s")
            st.metric("Est. Remaining", f"{remaining/60:.1f} min")
            
            st.markdown("---")
            st.markdown("### Label Distribution")
            stats, total_reviewed = DataManager.get_label_stats(current_processed_path)
            if total_reviewed > 0:
                for lbl, count in stats.items():
                    pct = (count / total_reviewed) * 100
                    st.write(f"**{lbl}**: {count} ({pct:.1f}%)")
            else:
                st.write("No labels yet.")
            
        else:
            st.write("Load data to start.")

        st.markdown("---")
        st.title("💾 Export")
        
        render_export_section(current_raw_path, current_processed_path)

def render_export_section(raw_path, processed_path):
    export_option = st.selectbox(
        "Export Type",
        ["Training Ready (Verified + Modified)", "Verified Only", "All Reviewed", "Full Dataset (Merged)"]
    )
    
    if st.button("Prepare Download", width="stretch"): 
        jsonl_str = ""
        file_name = "export.jsonl"
        
        if "Full Dataset" in export_option:
            df_exp = DataManager.get_overview_data(raw_path, processed_path)
            # Prioritize Reviewed Entry, fallback to Raw
            lines = []
            for _, row in df_exp.iterrows():
                entry = row.get("Reviewed_Entry")
                if pd.isna(entry) or not entry:
                     entry = row.get("Raw_Entry")
                if entry and not pd.isna(entry):
                    lines.append(json.dumps(entry))
            jsonl_str = "\n".join(lines)
            file_name = "arciva_qa_full_merged.jsonl"
            
        else:
            # Reviewed data based
            valid_stats = []
            if "Training Ready" in export_option:
                valid_stats = ["Verified", "Modified"]
                file_name = "arciva_qa_training_ready.jsonl"
            elif "Verified Only" in export_option:
                valid_stats = ["Verified"]
                file_name = "arciva_qa_verified_only.jsonl"
            elif "All Reviewed" in export_option:
                valid_stats = ["Verified", "Modified", "Discarded"] 
                file_name = "arciva_qa_all_reviews.jsonl"
                
            lines = []
            if os.path.exists(processed_path):
                with open(processed_path, 'r') as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            meta = d.get("review_metadata", {})
                            label = meta.get("label", "")
                            
                            if any(s in label for s in valid_stats):
                                lines.append(line.strip())
                        except: pass
            jsonl_str = "\n".join(lines)
            
        if jsonl_str:
            st.download_button(
                label="⬇️ Download JSONL",
                data=jsonl_str,
                file_name=file_name,
                mime="application/jsonl",
                width="stretch"
            )
            st.success(f"Prepared {len(lines)} items.")
        else:
            st.warning("No matching data found.")
