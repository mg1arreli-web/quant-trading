"""
Tests for utils/state_manager.py — atomic writes, corruption recovery.
"""
import os

from utils.state_manager import StateManager


class TestStateManager:
    def _make_manager(self, tmp_path):
        path = os.path.join(str(tmp_path), 'test_state.json')
        return StateManager(file_path=path)

    def test_save_and_load(self, tmp_path):
        sm = self._make_manager(tmp_path)
        sm.save_state({'key': 'value', 'nested': {'a': 1}})
        loaded = sm.load_state()
        assert loaded == {'key': 'value', 'nested': {'a': 1}}

    def test_load_nonexistent_returns_empty(self, tmp_path):
        sm = self._make_manager(tmp_path)
        assert sm.load_state() == {}

    def test_load_corrupted_returns_empty(self, tmp_path):
        sm = self._make_manager(tmp_path)
        os.makedirs(os.path.dirname(sm.file_path), exist_ok=True)
        with open(sm.file_path, 'w') as f:
            f.write('not valid json {{{}}}')
        assert sm.load_state() == {}

    def test_overwrite_state(self, tmp_path):
        sm = self._make_manager(tmp_path)
        sm.save_state({'v': 1})
        sm.save_state({'v': 2})
        assert sm.load_state() == {'v': 2}

    def test_save_creates_directory(self, tmp_path):
        deep_path = os.path.join(str(tmp_path), 'a', 'b', 'c', 'state.json')
        sm = StateManager(file_path=deep_path)
        sm.save_state({'deep': True})
        assert sm.load_state() == {'deep': True}
