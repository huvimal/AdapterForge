"""Optional integration smoke check using the actual saved model, never a mock."""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
from api import create_app

ROOT = Path(__file__).resolve().parent
TEXT = 'Please urgently track my shipment for ORD-3197.'


def main():
    app = create_app()
    with TestClient(app) as client:
        health = client.get('/health')
        assert health.status_code == 200 and health.json()['ready']
        response = client.post('/extract', json={'text': TEXT})
        assert response.status_code == 200, response.text
        result = response.json()
        assert 'raw_output' in result and 'validation' in result
        assert client.post('/extract', json={'text': ' '}).status_code == 422
    # This loads the actual run-v1 adapter in the Streamlit application as well.
    ui = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
    assert not ui.exception, list(ui.exception)
    next(s for s in ui.selectbox if s.label == 'Model').set_value('adapter').run()
    versions = next(s for s in ui.selectbox if s.label == 'Adapter version')
    versions.set_value(ROOT / 'private/run-v1/best').run()
    ui.text_area[0].set_value(TEXT)
    next(b for b in ui.button if b.label == 'Run local model').click().run(timeout=120)
    assert not ui.exception and not ui.error, (list(ui.exception), list(ui.error))
    assert any(item.value == result['raw_output'] for item in ui.text), 'UI/API raw output differs'
    output = {'scope': 'Actual local checkpoint, one request, no accuracy or load-test claim.',
              'request': TEXT, 'api_status': response.status_code, 'api_result': result,
              'streamlit_actual_model': 'same raw output; no application exception',
              'expected_by_task': {'intent': 'delivery', 'urgency': 'urgent', 'order_id': 'ORD-3197'}}
    path = ROOT / 'reports/local-smoke.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
