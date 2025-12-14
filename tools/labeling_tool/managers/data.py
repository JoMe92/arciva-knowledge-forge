
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
    def load_data(raw_path, processed_path, limit_pct=1.0):
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
        if os.path.exists(processed_path):
            with open(processed_path, 'r') as f:
                for line in f:
                    try:
                        rev_entry = json.loads(line)
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
            if len(unreviewed_data) > 0 and sample_size == 0:
                sample_size = 1
            sampled_data = random.sample(unreviewed_data, sample_size)
        else:
            sampled_data = random.sample(unreviewed_data, len(unreviewed_data)) 

        return sampled_data, len(data), len(reviewed_contents)

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
    def save_entry(entry, label, corrected_answer, original_entry, processed_path):
        """
        Appends the reviewed entry to the processed file.
        """
        entry["messages"][1]["content"] = corrected_answer
        
        entry["review_metadata"] = {
            "status": "checked",
            "label": label,
            "timestamp": time.time()
        }
        
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        
        with open(processed_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
