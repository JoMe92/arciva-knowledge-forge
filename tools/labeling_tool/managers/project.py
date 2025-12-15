
import os
import json
import time
import uuid
import shutil
from datetime import datetime

class ProjectManager:
    def __init__(self, filepath="tools/labeling_tool/db/projects.json"):
        self.filepath = filepath
        self.ensure_file_exists()

    def ensure_file_exists(self):
        dirname = os.path.dirname(self.filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({"projects": {}}, f, indent=4)

    def load_projects(self):
        with open(self.filepath, 'r') as f:
            try:
                data = json.load(f)
                return data.get("projects", {})
            except json.JSONDecodeError:
                return {}

    def save_projects(self, projects):
        with open(self.filepath, 'w') as f:
            json.dump({"projects": projects}, f, indent=4)

    def create_project(self, name, description, raw_path, processed_path):
        projects = self.load_projects()
        project_id = str(uuid.uuid4())
        
        # Enforce or suggest Data paths?
        # For now, we trust the defaults passed in or user inputs
        
        projects[project_id] = {
            "name": name,
            "description": description,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "raw_data_path": raw_path,
            "processed_data_path": processed_path,
            "status": "New"
        }
        self.save_projects(projects)
        return project_id

    def update_project(self, project_id, updates):
        projects = self.load_projects()
        if project_id in projects:
            projects[project_id].update(updates)
            projects[project_id]["last_accessed"] = time.time()
            self.save_projects(projects)
            return True
        return False

    def get_project(self, project_id):
        projects = self.load_projects()
        return projects.get(project_id)
    
    def delete_project(self, project_id):
        projects = self.load_projects()
        if project_id in projects:
            del projects[project_id]
            self.save_projects(projects)
            return True
        return False
