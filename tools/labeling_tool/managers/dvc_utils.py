import os
import yaml

def resolve_dvc_path(path: str) -> str:
    """
    Checks if the given path is a .dvc file. If so, parses it to find
    the tracked file path. Returns the absolute path to the actual data file.
    If not a .dvc file or parsing fails, returns the original path.
    """
    if path is None:
        return None
        
    if not path or not path.endswith('.dvc'):
        return path

    if not os.path.exists(path):
        return path

    try:
        with open(path, 'r') as f:
            content = yaml.safe_load(f)
        
        # Look for 'outs' which is a list of outputs
        outs = content.get('outs', [])
        if outs and isinstance(outs, list):
            # Usually the first output is what we want for a simple tracking file
            first_out = outs[0]
            tracked_filename = first_out.get('path')
            
            if tracked_filename:
                # The path in .dvc is relative to the .dvc file itself
                dir_name = os.path.dirname(path)
                real_path = os.path.join(dir_name, tracked_filename)
                return os.path.normpath(real_path)
                
    except Exception as e:
        print(f"Warning: Failed to resolve DVC path {path}: {e}")
        
    return path
