"""
State Manager — persistent JSON state with atomic writes.
"""
import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self, file_path: str = 'data/system_state.json'):
        self.file_path = file_path
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def load_state(self) -> dict:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load state from {self.file_path}: {e}")
            return {}

    def save_state(self, state: dict) -> None:
        """Atomic write: write to temp file, then os.replace() to target."""
        self._ensure_directory()
        directory = os.path.dirname(self.file_path) or '.'
        try:
            fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
            with os.fdopen(fd, 'w') as f:
                json.dump(state, f, indent=4)
            os.replace(tmp_path, self.file_path)
        except OSError as e:
            logger.error(f"Failed to save state: {e}")
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
