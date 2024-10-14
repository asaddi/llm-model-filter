from contextlib import asynccontextmanager
from enum import Enum
from pprint import pprint
from typing import Any, AsyncGenerator, List

from aiohttp import ClientResponse, ClientSession
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import yaml


class ObjectList(Enum):
    list = 'list'


class ObjectModel(Enum):
    model = 'model'


class Model(BaseModel):
    id: str = Field(
        ...,
        description='The model identifier, which can be referenced in the API endpoints.',
    )
    created: int = Field(
        ..., description='The Unix timestamp (in seconds) when the model was created.'
    )
    object: ObjectModel = Field(
        ..., description='The object type, which is always "model".'
    )
    owned_by: str = Field(..., description='The organization that owns the model.')


class ListModelsResponse(BaseModel):
    object: ObjectList
    data: List[Model]


TARGET = 'http://localhost:8080'


CLIENT: ClientSession|None = None


@asynccontextmanager
async def setup_teardown(_):
    global CLIENT

    CLIENT = ClientSession(base_url=TARGET)
    try:
        yield
    finally:
        await CLIENT.close()


app = FastAPI(title='OpenAI-compatible API model filter proxy', lifespan=setup_teardown)


async def LineCopyStreamer(resp: ClientResponse) -> AsyncGenerator[Any, Any]:
    try:
        while True:
            line = await resp.content.readline()
            if line == b'': break
            yield line
    finally:
        resp.close()


async def streaming_aware_proxy(request: Request, endpoint: str):
    """
    Proxy a potential SSE-streaming request to another URL.

    If 'stream' is `False` or not present in the request body, then a simple
    proxy request is performed (JSON-in -> JSON-out).

    Otherwise, Server Side Events are streamed line-by-line from target URL
    to client.
    """
    assert CLIENT is not None

    req_body = await request.json()
    pprint(req_body)

    resp = await CLIENT.request('POST', endpoint, json=req_body)

    if not req_body.get('stream', False):
        resp_json = await resp.json()
        resp.close()
        return resp_json

    return StreamingResponse(
        LineCopyStreamer(resp),
        media_type="text/event-stream",
    )


async def simple_proxy(request: Request, endpoint: str):
    """
    Proxy a simple request (JSON-in -> JSON-out) to another URL.
    """
    assert CLIENT is not None

    req_body = await request.json()
    pprint(req_body)

    async with CLIENT.request('POST', endpoint, json=req_body) as resp:
        return await resp.json()


@app.post('/v1/completions')
async def create_completion(request: Request):
    """
    Creates a completion for the provided prompt and parameters.
    """
    return await streaming_aware_proxy(request, '/v1/completions')


@app.post('/v1/chat/completions')
async def create_chat_completion(request: Request):
    """
    Creates a model response for the given chat conversation.
    """
    return await streaming_aware_proxy(request, '/v1/chat/completions')


@app.post('/v1/embeddings')
async def create_embedding(request: Request):
    """
    Creates an embedding vector representing the input text.
    """
    return await simple_proxy(request, '/v1/embeddings')


@app.get('/v1/models', response_model=ListModelsResponse)
async def list_models() -> ListModelsResponse:
    """
    Lists the currently available models, and provides basic information about each one such as the owner and availability.
    """
    assert CLIENT is not None

    async with CLIENT.request('GET', '/v1/models') as resp:
        resp_json = await resp.json()
    resp_models = ListModelsResponse.model_validate(resp_json)

    # TODO filter the models

    return resp_models
