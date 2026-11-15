import json
import hashlib
from pathlib import Path
import threading
import streamlit as st

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def engine(variant, adapter_path):
    from runtime import load_base, load_adapter
    tokenizer, model, device = load_base(offline=True) if variant == 'base' else load_adapter(adapter_path, offline=True)
    return tokenizer, model, device, threading.Lock()


def main():
    st.set_page_config(page_title='AdapterForge', layout='wide')
    st.title('AdapterForge')
    st.caption('Actual LoRA fine-tuning → frozen evaluation → local model inference')
    st.info('English synthetic support requests. No external inference API. Raw model output is validated, not repaired. Run train.py before selecting the adapter.')
    versions = sorted(p for p in (ROOT / 'private').glob('run-*/best') if (p / 'adapter_model.safetensors').exists() and (p.parent / 'training.json').exists())
    variant = st.selectbox('Model', ['adapter', 'base'] if versions else ['base'])
    adapter_path = st.selectbox('Adapter version', versions, format_func=lambda p: p.parent.name) if variant == 'adapter' else None
    report_path = ROOT / 'reports/comparison.json'
    if report_path.exists():
        with st.expander('Measured base / adapter / rule comparison', expanded=True):
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if adapter_path and hashlib.sha256((adapter_path / 'adapter_model.safetensors').read_bytes()).hexdigest() != report['adapter_sha256']:
                st.warning('This report belongs to another adapter. Re-evaluate the selected version before comparing metrics.')
            for split, variants in report['evaluation'].items():
                st.subheader(split)
                st.dataframe([{'variant': name, **{key: v[key] for key in ('cases', 'raw_json_valid_rate', 'schema_valid_rate', 'exact_match')}} for name, v in variants.items()], hide_index=True)
            st.caption('These are actual local results on synthetic held-out templates and authored challenges. Not real-client or production accuracy.')
    text = st.text_area('English request', 'Please check the delivery status for ORD-8765. This is not urgent.', max_chars=1200)
    if st.button('Run local model'):
        try:
            from runtime import infer
            with st.spinner('Loading/running local model...'):
                tokenizer, model, device, lock = engine(variant, adapter_path)
                with lock:
                    result = infer(tokenizer, model, text)
            st.text(result['raw_output'])
            st.json(result['validation'])
            st.caption(f"Device: {device}; generation {result['seconds']:.3f} seconds, excluding load time.")
            if not result['validation']['schema_valid'] or not result['validation']['grounded']:
                st.warning('Output is not usable without review: invalid schema or unsupported order ID. No regex repair or fallback is applied.')
        except Exception as exc:
            st.error(f'Local inference unavailable: {type(exc).__name__}. Check the trained adapter, model cache and console. No input is sent externally.')


if __name__ == '__main__':
    main()
