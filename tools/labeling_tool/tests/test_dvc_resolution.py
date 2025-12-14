
import os
import sys
import unittest
import shutil
import tempfile
import yaml

# Add the tools/labeling_tool directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
labeling_tool_dir = os.path.dirname(current_dir)
sys.path.insert(0, labeling_tool_dir)

from managers.dvc_utils import resolve_dvc_path
from managers.data import DataManager

class TestDVCResolution(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.test_dir, "data.jsonl")
        self.dvc_file = os.path.join(self.test_dir, "data.jsonl.dvc")
        
        # Create dummy data file
        with open(self.data_file, 'w') as f:
            f.write('{"instruction": "test", "output": "test"}\n')
            
        # Create dummy dvc file
        dvc_content = {
            "outs": [
                {
                    "md5": "12345",
                    "path": "data.jsonl"
                }
            ]
        }
        with open(self.dvc_file, 'w') as f:
            yaml.dump(dvc_content, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_resolve_dvc_path_exists(self):
        resolved = resolve_dvc_path(self.dvc_file)
        self.assertEqual(resolved, self.data_file)

    def test_resolve_dvc_path_not_exists(self):
        # A file that behaves like a dvc file but points to dvc file that doesn't exist
        fake_path = os.path.join(self.test_dir, "fake.dvc")
        resolved = resolve_dvc_path(fake_path)
        self.assertEqual(resolved, fake_path)

    def test_resolve_normal_file(self):
        resolved = resolve_dvc_path(self.data_file)
        self.assertEqual(resolved, self.data_file)

    def test_data_manager_load(self):
        # Test loading via DataManager with DVC path
        data, total, reviewed = DataManager.load_data(self.dvc_file, "dummy_processed_path")
        self.assertEqual(len(data), 1)
        self.assertEqual(total, 1)

if __name__ == '__main__':
    unittest.main()
