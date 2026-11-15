"""Loopback-oriented demo service. No auth or public deployment claim."""
from contextlib import asynccontextmanager
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=1, max_length=1200)

    @field_validator('text')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Text cannot be blank')
        return value


def create_app(engine=None):
    @asynccontextmanager
    async def lifespan(app):
        if engine is None:
            from runtime import load_adapter, infer
            tokenizer, model, device = load_adapter('private/run-v1/best', offline=True)
            app.state.engine = lambda text: infer(tokenizer, model, text)
        yield
    app = FastAPI(title='AdapterForge', lifespan=lifespan)
    app.state.engine = engine
    lock = threading.Lock()

    @app.get('/health')
    def health():
        return {'ready': app.state.engine is not None, 'scope': 'local demo'}

    @app.post('/extract')
    def extract(request: Request):
        if app.state.engine is None or not lock.acquire(blocking=False):
            raise HTTPException(503, 'Model unavailable or busy; retry later')
        try:
            return app.state.engine(request.text)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        finally:
            lock.release()
    return app


app = create_app()
