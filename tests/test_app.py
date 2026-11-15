import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class UiTests(unittest.TestCase):
    def test_raw_invalid_model_output_is_not_repaired(self):
        fake = {'raw_output': 'not json', 'validation': {'json_valid': False, 'schema_valid': False, 'grounded': False, 'value': None}, 'seconds': .1}
        with patch('runtime.load_base', return_value=(object(), object(), 'cpu')), patch('runtime.infer', return_value=fake):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
            next(s for s in app.selectbox if s.label == 'Model').set_value('base').run()
            next(b for b in app.button if b.label == 'Run local model').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('not usable' in item.value for item in app.warning))
