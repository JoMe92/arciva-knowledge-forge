
import os
import json
import random
import pandas as pd
import time

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
        data = []
        if not os.path.exists(raw_path):
            return [], 0, 0

        # 1. Load all raw data
        with open(raw_path, 'r') as f:
            for line in f:
                try:
                    line_data = json.loads(line)
                    formatted = DataManager.ensure_messages_format(line_data)
                    if formatted:
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
        raw_items = []
        if os.path.exists(raw_path):
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
        if os.path.exists(processed_path):
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
        Appends the reviewed entry to the processed file, converting format if needed.
        """
        # Update content first (in standard multi-turn structure)
        entry["messages"][1]["content"] = corrected_answer
        
        # Add metadata
        entry["review_metadata"] = {
            "status": "checked",
            "label": label,
            "timestamp": time.time(),
            "format": save_format
        }
        
        # Convert if needed
        output_entry = entry
        if save_format == "Alpaca":
            user_msg = next((m["content"] for m in entry["messages"] if m["role"] == "user"), "")
            asst_msg = entry["messages"][1]["content"] # Already updated above
            output_entry = {
                "instruction": user_msg,
                "input": "",
                "output": asst_msg,
                "review_metadata": entry["review_metadata"],
                "original_meta": entry.get("original_meta", {})
            }
        
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        
        with open(processed_path, 'a') as f:
            f.write(json.dumps(output_entry) + "\n")
