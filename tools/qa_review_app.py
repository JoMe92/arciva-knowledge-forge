
import streamlit as st
import pandas as pd
import json
import random
import os
import time

# --- Configuration ---
RAW_DATA_PATH = "data/raw/arciva_qa_synthetic.jsonl"
PROCESSED_DATA_PATH = "data/processed/arciva_qa_reviewed.jsonl" # Target for reviews

# --- Helper Functions ---

def ensure_messages_format(entry):
    """
    Converts 'instruction'/'output' format to 'messages' format if needed.
    """
    if "messages" in entry:
        return entry
    
    if "instruction" in entry and "output" in entry:
        return {
            "messages": [
                {"role": "user", "content": entry["instruction"]},
                {"role": "assistant", "content": entry["output"]}
            ],
            "original_meta": {k:v for k,v in entry.items() if k not in ["instruction", "output"]}
        }
    return None # Invalid format

def load_data(filepath, limit_pct=1.0):
    """
    Loads data, excludes already reviewed items, and samples based on percentage.
    """
    data = []
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        return []

    # 1. Load all raw data
    with open(filepath, 'r') as f:
        for line in f:
            try:
                line_data = json.loads(line)
                formatted = ensure_messages_format(line_data)
                if formatted:
                    # Create a simple hash/ID for deduplication if needed
                    # For now, we trust the random sampling + exclusion of exact content matches
                    # But simpler: just append "id" based on index if no ID exists? 
                    # Actually, raw data might change line numbers. 
                    # Let's rely on content hashing or just list exclusion?
                    # REQUIREMENT: "Data pairs reviewed... must not reappear".
                    # Let's simple check: content of user message.
                    data.append(formatted)
            except json.JSONDecodeError:
                continue
    
    # 2. Load already reviewed data to exclude
    reviewed_contents = set()
    if os.path.exists(PROCESSED_DATA_PATH):
        with open(PROCESSED_DATA_PATH, 'r') as f:
            for line in f:
                try:
                    rev_entry = json.loads(line)
                    # Assuming we check the user message content for uniqueness
                    msgs = rev_entry.get("messages", [])
                    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
                    if user_msg:
                        reviewed_contents.add(user_msg)
                except:
                    pass
    
    # 3. Filter
    unreviewed_data = [d for d in data if d["messages"][0]["content"] not in reviewed_contents]
    
    # 4. Sample
    if limit_pct < 1.0:
        sample_size = int(len(unreviewed_data) * limit_pct)
        # Ensure at least 1 if data exists but sample triggers 0
        if len(unreviewed_data) > 0 and sample_size == 0:
            sample_size = 1
        
        # User requirement: Randomization
        # We sample randomly from the available pool
        sampled_data = random.sample(unreviewed_data, sample_size)
    else:
        sampled_data = random.sample(unreviewed_data, len(unreviewed_data)) # Still shuffle for "Randomization" requirement

    return sampled_data, len(data), len(reviewed_contents)

def get_label_stats():
    """Calculates label distribution from the processed file."""
    stats = {"✅ Verified": 0, "✍️ Modified": 0, "🗑️ Discarded": 0}
    total = 0
    if os.path.exists(PROCESSED_DATA_PATH):
        with open(PROCESSED_DATA_PATH, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    meta = entry.get("review_metadata", {})
                    lbl = meta.get("label", "")
                    
                    if "Verified" in lbl or "Correct" in lbl: stats["✅ Verified"] += 1
                    elif "Modified" in lbl: stats["✍️ Modified"] += 1
                    elif "Discarded" in lbl: stats["🗑️ Discarded"] += 1
                    # Legacy mapping if needed, or just let them fall through/count as other?
                    # For now, let's keep it simple.
                    
                    total += 1
                except:
                    pass
    return stats, total

def get_overview_data(raw_path, processed_path):
    """
    Merges raw and processed data to pivot a full status view.
    """
    # 1. Load Raw
    raw_items = []
    if os.path.exists(raw_path):
        with open(raw_path, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    fmt = ensure_messages_format(d)
                    if fmt:
                        q = next((m["content"] for m in fmt["messages"] if m["role"] == "user"), "")
                        raw_items.append({"Question": q, "Raw_Entry": fmt})
                except: pass
    
    df_raw = pd.DataFrame(raw_items)
    
    # 2. Load Processed
    proc_items = []
    if os.path.exists(processed_path):
        with open(processed_path, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    q = next((m["content"] for m in d["messages"] if m["role"] == "user"), "")
                    meta = d.get("review_metadata", {})
                    status = meta.get("label", "⚠️ Unknown") # e.g. "✅ Verified"
                    proc_items.append({"Question": q, "Status": status, "Reviewed_Entry": d})
                except: pass
                
    df_proc = pd.DataFrame(proc_items)
    
    # 3. Merge
    if not df_raw.empty:
        if not df_proc.empty:
            # Drop duplicates in processed if any (take latest)
             df_proc = df_proc.drop_duplicates(subset=["Question"], keep="last")
             df_merged = pd.merge(df_raw, df_proc, on="Question", how="left")
        else:
            df_merged = df_raw
            df_merged["Status"] = None
            
        # Fill Pending
        df_merged["Status"] = df_merged["Status"].fillna("⏳ Pending")
        return df_merged
    return pd.DataFrame()

def save_entry(entry, label, corrected_answer, original_entry):
    """
    Appends the reviewed entry to the processed file.
    """
    # specific logic to update the assistant message
    entry["messages"][1]["content"] = corrected_answer
    
    # Add metadata
    entry["review_metadata"] = {
        "status": "checked",
        "label": label,
        "timestamp": time.time()
    }
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    with open(PROCESSED_DATA_PATH, 'a') as f:
        f.write(json.dumps(entry) + "\n")

# --- UI Setup ---
st.set_page_config(page_title="Arciva QA Review", layout="wide")

# Initialize Session State
if 'data_queue' not in st.session_state:
    st.session_state['data_queue'] = []
if 'current_index' not in st.session_state:
    st.session_state['current_index'] = 0
if 'reviews_this_session' not in st.session_state:
    st.session_state['reviews_this_session'] = 0
if 'session_start_time' not in st.session_state:
    st.session_state['session_start_time'] = time.time()

# --- SIDEBAR: Config & Dashboard ---
with st.sidebar:
    st.title("⚙️ Config")
    
    # Sampling Controls
    # Only show if queue is empty or reset requested
    if len(st.session_state['data_queue']) == 0:
        input_path = st.text_input("Input File Path", value=RAW_DATA_PATH)
        
        sample_pct = st.slider("Sample Size (%)", 1, 100, 10)
        full_dataset = st.checkbox("Full Dataset (100%)")
        if full_dataset:
            sample_pct = 100
        
        if st.button("Load Session"):
            queue, total_avail, total_reviewed = load_data(input_path, sample_pct/100.0)
            st.session_state['data_queue'] = queue
            st.session_state['total_avail_count'] = total_avail # Snapshot for stats
            st.session_state['total_reviewed_history'] = total_reviewed
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
        # We could count active labels in session state if we wanted "Real-time label distribution"
        # For simplicity, we track simple session count
        elapsed = time.time() - st.session_state['session_start_time']
        avg_time = elapsed / max(1, st.session_state['reviews_this_session'])
        remaining = (queue_len - current) * avg_time
        
        st.metric("Avg Time / Pair", f"{avg_time:.1f}s")
        st.metric("Est. Remaining", f"{remaining/60:.1f} min")
        
        st.markdown("---")
        st.markdown("### Label Distribution")
        stats, total_reviewed = get_label_stats()
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
    
    export_option = st.selectbox(
        "Export Type",
        ["Training Ready (Verified + Modified)", "Verified Only", "All Reviewed", "Full Dataset (Merged)"]
    )
    
    if st.button("Prepare Download", width="stretch"): # Replaced deprecated arg
        # Logic to prepare data
        jsonl_str = ""
        file_name = "export.jsonl"
        
        if "Full Dataset" in export_option:
            df_exp = get_overview_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
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
                valid_stats = ["Verified", "Modified", "Discarded"] # All types
                file_name = "arciva_qa_all_reviews.jsonl"
                
            lines = []
            if os.path.exists(PROCESSED_DATA_PATH):
                with open(PROCESSED_DATA_PATH, 'r') as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            meta = d.get("review_metadata", {})
                            label = meta.get("label", "")
                            
                            # Check if label matches any of our valid stats keywords
                            if any(s in label for s in valid_stats):
                                lines.append(line.strip())
                        except: pass
            jsonl_str = "\n".join(lines)
            
        # Download Button
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

# --- MAIN AREA ---
# View Switching
mode = st.sidebar.radio("View Mode", ["Review", "Overview"], index=0)

st.title("Arciva Dataset Review Tool")

if mode == "Overview":
    st.header("Dataset Overview")
    
    input_p = st.session_state.get('input_path_cache', RAW_DATA_PATH) 
    # Use cached input path from session if available (though st.text_input in sidebar usually drives it)
    # Actually, we need to read the current text_input value. 
    # But sidebar is rendered above. Let's just use the variable we defined if possible, 
    # or re-read from session state if keys match.
    # Limitation: input_path variable scope.
    # Fix: Store input_path in session state in the sidebar callback or just access it if key is set?
    # Better: Re-read the default. Or better, `load_data` updates a session var.
    
    df = get_overview_data(RAW_DATA_PATH, PROCESSED_DATA_PATH) # Simplified: Using constants for now or we need to wire up the input
    
    if not df.empty:
        # Stats
        st.markdown("### Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pairs", len(df))
        c2.metric("Verified", len(df[df["Status"].str.contains("Verified")]))
        c3.metric("Modified", len(df[df["Status"].str.contains("Modified")]))
        c4.metric("Pending", len(df[df["Status"].str.contains("Pending")]))
        
        # Color Styling
        def highlight_status(val):
            color = ''
            if 'Verified' in val: color = 'background-color: #d4edda; color: #155724' # Green
            elif 'Modified' in val: color = 'background-color: #cce5ff; color: #004085' # Blue
            elif 'Discarded' in val: color = 'background-color: #f8d7da; color: #721c24' # Red
            elif 'Pending' in val: color = 'background-color: #fff3cd; color: #856404' # Yellow
            return color

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
            # Flexible filtering matches partial string (e.g. "Verified" matches "✅ Verified")
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
        ) # styling via pandas styler is possible but `st.dataframe` is simpler for now. 
        # Advanced: st.dataframe(df.style.map(highlight_status, subset=['Status'])) 
        # But let's stick to simple first to ensure robustness.
        
    else:
        st.info("No data found.")

elif mode == "Review":
    if len(st.session_state['data_queue']) > 0 and st.session_state['current_index'] < len(st.session_state['data_queue']):
        
        # Get current item
        item = st.session_state['data_queue'][st.session_state['current_index']]
        
        # Extract Q & A
        # Assuming standard structure [User, Assistant]
        user_msg = next((m for m in item["messages"] if m["role"] == "user"), None)
        asst_msg = next((m for m in item["messages"] if m["role"] == "assistant"), None)
        
        if not user_msg or not asst_msg:
            st.error("Invalid message format in current item.")
            # helper to skip
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
                    key=f"answer_{st.session_state['current_index']}", # Unique key per item
                    on_change=lambda: None # Force rerun on blur
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
            
            # Discard Logic
            with col1:
                if st.button("🗑️ Discard Pair", type="secondary", width="stretch"):
                     save_entry(item, "🗑️ Discarded", new_answer, item) # Save as discarded
                     st.session_state['current_index'] += 1
                     st.session_state['reviews_this_session'] += 1
                     st.rerun()
    
            # Save/Next Logic
            with col2:
                # We combine "Next" and "Save" into one primary action
                # Requirement: "The user should also have the option to close pairs completely" -> Done via Discard/Save
                # "that should then also be tracked" -> Done via save_entry
                
                if st.button("Confirm & Next ➡️", type="primary", width="stretch"):
                    save_entry(item, current_status, new_answer, item)
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

