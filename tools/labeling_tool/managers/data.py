
import os
import json
import random
import pandas as pd
import time
from .dvc_utils import resolve_dvc_path

class DataManager:
    @staticmethod
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
        return None 

    @staticmethod
    def load_data(raw_path, processed_path, limit_pct=1.0, include_reviewed=False):
        """
        Loads data, excludes already reviewed items, and samples based on percentage.
        """
        # Resolve DVC paths if necessary
        raw_path = resolve_dvc_path(raw_path)
        processed_path = resolve_dvc_path(processed_path)
        
        # Fallback: If processed path is missing, we assume "edit in place" logic 
        # but for loading "already reviewed", if it's the SAME file, we need to be careful.
        # However, typically processed_path should be set to raw_path in the project config if it was empty.
        # If it is still None here, treat it as "no separate processed file" (so effective processed path is raw_path)
        if not processed_path and raw_path:
            processed_path = raw_path

        data = []
        if not raw_path or not os.path.exists(raw_path):
            return [], 0, 0


        # 1. Load all raw data
        with open(raw_path, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    line_data = json.loads(line)
                    # Detect original format
                    orig_fmt = "alpaca" if "instruction" in line_data and "output" in line_data else "messages"
                    
                    formatted = DataManager.ensure_messages_format(line_data)
                    if formatted:
                        # Store internal metadata for in-place editing
                        formatted["_line_index"] = idx
                        formatted["_original_format"] = orig_fmt
                        data.append(formatted)
                except json.JSONDecodeError:
                    continue
        
        # 2. Load already reviewed data to exclude
        reviewed_contents = set()
        reviewed_items = []
        if os.path.exists(processed_path):
            with open(processed_path, 'r') as f:
                for line in f:
                    try:
                        rev_entry = json.loads(line)
                        msgs = rev_entry.get("messages", [])
                        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
                        if user_msg:
                            reviewed_contents.add(user_msg)
                            if include_reviewed:
                                reviewed_items.append(rev_entry)
                    except:
                        pass
        
        # 3. Filter
        unreviewed_data = [d for d in data if d["messages"][0]["content"] not in reviewed_contents]
        
        # 4. Calculate Sampling
        sampled_data = []

        if include_reviewed:
            # Inclusive logic: "10%" means "10% of total data should be in the session".
            # The session will contain ALL reviewed data + (Target - Reviewed) new data.
            # If Reviewed > Target, we still return all reviewed (or should we cap it? User said "all previously labeled... included").
            # Let's pivot to: The slider is "Additional New Data" or "Total Session Size"?
            # User said: "die 10 sollen inkulsive ebreties gelabelter daten sien" -> "The 10 [%] should be inclusive of already labeled data".
            
            total_count_target = int(len(data) * limit_pct)
            current_reviewed_count = len(reviewed_items)
            needed_new_count = total_count_target - current_reviewed_count
            
            if needed_new_count > 0:
                safe_count = min(needed_new_count, len(unreviewed_data))
                sampled_data = random.sample(unreviewed_data, safe_count)
            else:
                sampled_data = [] # Target already met/exceeded by reviewed items
                
            final_queue = reviewed_items + sampled_data
            
        else:
            # Exclusive logic (Default): "10%" means "10% of unreviewed data".
            if limit_pct < 1.0:
                sample_size = int(len(unreviewed_data) * limit_pct)
                if len(unreviewed_data) > 0 and sample_size == 0:
                    sample_size = 1
                sampled_data = random.sample(unreviewed_data, sample_size)
            else:
                sampled_data = random.sample(unreviewed_data, len(unreviewed_data)) 
            
            final_queue = sampled_data

        return final_queue, len(data), len(reviewed_contents)

    @staticmethod
    def get_label_stats(processed_path):
        """Calculates label distribution from the processed file."""
        processed_path = resolve_dvc_path(processed_path)
        if not processed_path:
             return {"✅ Verified": 0, "✍️ Modified": 0, "🗑️ Discarded": 0}, 0

        stats = {"✅ Verified": 0, "✍️ Modified": 0, "🗑️ Discarded": 0}
        total = 0
        if os.path.exists(processed_path):
            with open(processed_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        meta = entry.get("review_metadata", {})
                        lbl = meta.get("label", "")
                        
                        if "Verified" in lbl or "Correct" in lbl: stats["✅ Verified"] += 1
                        elif "Modified" in lbl: stats["✍️ Modified"] += 1
                        elif "Discarded" in lbl: stats["🗑️ Discarded"] += 1
                        
                        total += 1
                    except:
                        pass
        return stats, total

    @staticmethod
    def get_overview_data(raw_path, processed_path):
        """
        Merges raw and processed data to pivot a full status view.
        """
        raw_path = resolve_dvc_path(raw_path)
        processed_path = resolve_dvc_path(processed_path)
        
        # Fallback for edit-in-place
        if not processed_path:
            processed_path = raw_path

        raw_items = []
        if raw_path and os.path.exists(raw_path):
            with open(raw_path, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        fmt = DataManager.ensure_messages_format(d)
                        if fmt:
                            q = next((m["content"] for m in fmt["messages"] if m["role"] == "user"), "")
                            raw_items.append({"Question": q, "Raw_Entry": fmt})
                    except: pass
        
        df_raw = pd.DataFrame(raw_items)
        
        proc_items = []
        if processed_path and os.path.exists(processed_path):
            with open(processed_path, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        q = next((m["content"] for m in d["messages"] if m["role"] == "user"), "")
                        meta = d.get("review_metadata", {})
                        status = meta.get("label", "⚠️ Unknown") 
                        proc_items.append({"Question": q, "Status": status, "Reviewed_Entry": d})
                    except: pass
                    
        df_proc = pd.DataFrame(proc_items)
        
        if not df_raw.empty:
            if not df_proc.empty:
                df_proc = df_proc.drop_duplicates(subset=["Question"], keep="last")
                df_merged = pd.merge(df_raw, df_proc, on="Question", how="left")
            else:
                df_merged = df_raw
                df_merged["Status"] = None
                
            df_merged["Status"] = df_merged["Status"].fillna("⏳ Pending")
            return df_merged
        return pd.DataFrame()

    @staticmethod
    def save_entry(entry, label, corrected_answer, original_entry, processed_path, save_format="Multi-turn Dialog"):
        """
        Updates the entry IN-PLACE in the processed file (which is often the raw file).
        Preserves original schema (Alpaca vs Messages).
        """
        processed_path = resolve_dvc_path(processed_path)
        if not processed_path:
             raise ValueError("No processed path available for saving.")
        
        # 1. Update content in the working 'messages' format
        entry["messages"][1]["content"] = corrected_answer
        
        # 2. Add metadata
        entry["review_metadata"] = {
            "status": "checked",
            "label": label,
            "timestamp": time.time(),
            "format": save_format
        }
        
        # 3. Determine output format based on loaded metadata
        # If we know the original format was 'alpaca', we try to convert back to respect it.
        # Otherwise fall back to 'messages' or the requested save_format.
        orig_fmt = entry.get("_original_format", "messages")
        
        output_entry = entry
        if orig_fmt == "alpaca":
            user_msg = next((m["content"] for m in entry["messages"] if m["role"] == "user"), "")
            asst_msg = entry["messages"][1]["content"] 
            output_entry = {
                "instruction": user_msg,
                "input": "", # We might lose input if not preserved in original meta, but usually covered
                "output": asst_msg,
                "review_metadata": entry["review_metadata"],
            }
            # Merge back any other original keys safely
            orig_meta = entry.get("original_meta", {})
            for k, v in orig_meta.items():
                if k not in output_entry:
                    output_entry[k] = v
        else:
            # Default/Messages format
            # ensure we don't save our internal underscored keys
            output_entry = {k:v for k,v in entry.items() if not k.startswith('_')}

        # 4. In-Place Write
        # We need to read all lines, replace the specific line, and write back.
        if not os.path.exists(processed_path):
             # Should not happen for in-place edit of existing file
             with open(processed_path, 'a') as f:
                f.write(json.dumps(output_entry) + "\n")
             return

        # Read all
        with open(processed_path, 'r') as f:
            lines = f.readlines()
            
        target_idx = entry.get("_line_index")
        
        if target_idx is not None and 0 <= target_idx < len(lines):
            # Verify we are overwriting the correct thing (optional: regex check or partial match)
            # For now trust the index since we just loaded it.
            lines[target_idx] = json.dumps(output_entry) + "\n"
        else:
            # Fallback if index missing or out of bounds (e.g. file changed externally): Append
            lines.append(json.dumps(output_entry) + "\n")
            
        # Write all back
        with open(processed_path, 'w') as f:
            f.writelines(lines)
