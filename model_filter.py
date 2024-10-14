from contextlib import asynccontextmanager
from enum import Enum
import os
# from pprint import pprint
import re
from typing import Any, AsyncGenerator, List, Optional

from aiohttp import ClientResponse, ClientSession
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import yaml


class ModelFilterConfig(BaseModel):
    base_url: str
    api_key: Optional[str]=None
    case_sensitive: Optional[bool]=False
    regexp: Optional[list[str]]=None
    simple: Optional[list[str]]=None


# pydantic models for OpenAI API
# Since we're only parsing/modifying /v1/models, we don't care about all the
# other models (ChatCompletionRequest, etc.)

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


# Our global variables
CONFIG: ModelFilterConfig|None = None
CLIENT: ClientSession|None = None


@asynccontextmanager
async def setup_teardown(_):
    global CONFIG, CLIENT

    config_file = os.getenv('CONFIG_FILE', 'config.yaml')

    if config_file == 'config.yaml' and not os.path.exists(config_file):
        config_file = 'config.yaml.default'

    with open(config_file) as inp:
        config = yaml.load(inp, yaml.Loader)

    CONFIG = ModelFilterConfig.model_validate(config['model_filter'])
    # pprint(CONFIG)

    CONFIG.base_url = CONFIG.base_url.rstrip('/')

    # Resolve api_key from environment variable, if necessary
    api_key = CONFIG.api_key
    if api_key is not None:
        prefix = 'os.environ/'
        if api_key.startswith(prefix):
            api_key = os.environ[api_key[prefix:]]

    headers = None if api_key is None else \
        { 'Authorization': f'Bearer {api_key}'}
    CLIENT = ClientSession(headers=headers)
    try:
        yield
    finally:
        await CLIENT.close()


app = FastAPI(title='OpenAI-compatible API model filter proxy', lifespan=setup_teardown)


async def LineCopyStreamer(resp: ClientResponse) -> AsyncGenerator[Any, Any]:
    """
    Simply yields HTTP response one line at a time. Should work with SSE.
    """
    try:
        while True:
            line = await resp.content.readline()
            if line == b'': break
            yield line
    finally:
        resp.close()


def resolve_endpoint(endpoint: str) -> str:
    assert CONFIG is not None

    # Can't use urljoin because it gets rid of everything after the host/port.
    # Need to do a simple append.
    # Need to right-strip slashes elsewhere.
    return CONFIG.base_url+endpoint


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
    # pprint(req_body)

    resp = await CLIENT.request('POST', resolve_endpoint(endpoint), json=req_body)

    if not req_body.get('stream', False):
        # No streaming. Wait for JSON response and be on our way.
        resp_json = await resp.json()
        resp.close()
        return resp_json

    return StreamingResponse(
        LineCopyStreamer(resp), # NB This will close resp once done
        media_type="text/event-stream",
    )


async def simple_proxy(request: Request, endpoint: str):
    """
    Proxy a simple request (JSON-in -> JSON-out) to another URL.
    """
    assert CLIENT is not None

    req_body = await request.json()
    # pprint(req_body)

    async with CLIENT.request('POST', resolve_endpoint(endpoint), json=req_body) as resp:
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


def model_selected(model: Model) -> bool:
    """
    Returns `True` if the model is selected by our configuration.
    """
    assert CONFIG is not None

    # If the user didn't configure anything, pass it all
    if CONFIG.regexp is None and CONFIG.simple is None:
        return True

    mid = model.id

    if not CONFIG.case_sensitive:
        mid = mid.lower()

    if CONFIG.simple is not None:
        # TODO Cache?
        simple_list = CONFIG.simple if CONFIG.case_sensitive else \
            [ s.lower() for s in CONFIG.simple ]

        if mid in simple_list:
            return True

    # TODO optimize this, maybe by caching compiled REs?
    # Or does it matter?
    if CONFIG.regexp is not None:
        flags = 0 if CONFIG.case_sensitive else re.IGNORECASE
        for rexp in CONFIG.regexp:
            if re.fullmatch(rexp, mid, flags=flags) is not None:
                return True

    return False


@app.get('/v1/models', response_model=ListModelsResponse)
async def list_models() -> ListModelsResponse:
    """
    Lists the currently available models, and provides basic information about each one such as the owner and availability.
    """
    assert CLIENT is not None

    async with CLIENT.request('GET', resolve_endpoint('/v1/models')) as resp:
        resp_json = await resp.json()
    resp_models = ListModelsResponse.model_validate(resp_json)

    # Filter the models
    filtered_models = []
    for m in resp_models.data:
        if model_selected(m):
            filtered_models.append(m)

    return ListModelsResponse(
        object=ObjectList.list,
        data=filtered_models,
    )

# TODO Do typical frontends use the other model endpoints?
