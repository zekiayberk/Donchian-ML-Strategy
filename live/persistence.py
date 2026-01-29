import json
import os
import logging
from pathlib import Path

logger = logging.getLogger('StateStore')

class StateStore:
    def __init__(self, filepath='paper_state.json'):
        self.filepath = Path(filepath)
        
    def save(self, state):
        """
        Atomically saves state to JSON file.
        Writes to a tmp file first, then renames it to ensure data integrity.
        """
        try:
            tmp_path = self.filepath.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            # Atomic replace
            os.replace(tmp_path, self.filepath)
            # logger.debug(f"State saved to {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load(self):
        """
        Loads state from JSON file.
        Returns None if file doesn't exist or is corrupted.
        """
        if not self.filepath.exists():
            return None
            
        try:
            with open(self.filepath, 'r') as f:
                state = json.load(f)
            logger.info(f"State loaded from {self.filepath}")
            return state
        except json.JSONDecodeError:
            logger.error(f"State file corrupted: {self.filepath}")
            return None
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
