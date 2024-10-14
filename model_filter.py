# Copyright 2024 Allan Saddi <allan@saddi.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from contextlib import asynccontextmanager
import datetime
from enum import Enum
import logging
import os
# from pprint import pprint
import re
from typing import Annotated, Any, AsyncGenerator, List, Optional

from aiohttp import ClientResponse, ClientSession
from fastapi import FastAPI, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import yaml


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model_filter')


class ModelFilterConfig(BaseModel):
    base_url: str
    cache_ttl: Optional[int]=None
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
        default=ObjectModel.model, description='The object type, which is always "model".'
    )
    owned_by: str = Field(default='unknown', description='The organization that owns the model.')


class ListModelsResponse(BaseModel):
    object: ObjectList = ObjectList.list
    data: List[Model]


DEFAULT_CONFIG = 'config.yaml'


# Our global variables
CONFIG: ModelFilterConfig|None = None
CLIENT: ClientSession|None = None
SIMPLE_FILTERS: list[str] = []
RE_FILTERS: list[re.Pattern] = []


@asynccontextmanager
async def setup_teardown(_):
    global CONFIG, CLIENT
    global SIMPLE_FILTERS, RE_FILTERS

    config_file = os.getenv('LLM_MF_CONFIG', DEFAULT_CONFIG)

    if config_file == DEFAULT_CONFIG and not os.path.exists(config_file):
        config_file = os.path.join(os.path.dirname(__file__), 'config.yaml.default')

    logger.info(f'Reading configuration from {config_file}')

    with open(config_file) as inp:
        config = yaml.load(inp, yaml.Loader)

    CONFIG = ModelFilterConfig.model_validate(config['model_filter'])
    # pprint(CONFIG)

    CONFIG.base_url = CONFIG.base_url.rstrip('/')

    if CONFIG.simple is not None:
        SIMPLE_FILTERS = CONFIG.simple if CONFIG.case_sensitive else \
            [s.lower() for s in CONFIG.simple]

    if CONFIG.regexp is not None:
        flags = 0 if CONFIG.case_sensitive else re.IGNORECASE
        RE_FILTERS = [re.compile(p, flags=flags) for p in CONFIG.regexp]

    CLIENT = ClientSession()
    try:
        yield
    finally:
        await CLIENT.close()


# The main application object
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


def resolve_authorization(authorization: str|None) -> dict|None:
    return None if authorization is None else \
        { 'Authorization': authorization }


def resolve_endpoint(endpoint: str) -> str:
    assert CONFIG is not None

    # Can't use urljoin because it gets rid of everything after the host/port.
    # Need to do a simple append.
    # Need to right-strip slashes elsewhere.
    return CONFIG.base_url+endpoint


async def streaming_aware_proxy(request: Request, endpoint: str, authorization: str|None=None):
    """
    Proxy a potential SSE-streaming request to another URL.

    If 'stream' is `False` or not present in the request body, then a simple
    proxy request is performed (JSON-in -> JSON-out).

    Otherwise, Server Sent Events are streamed line-by-line from target URL
    to client.
    """
    assert CLIENT is not None

    req_body = await request.json()
    # pprint(req_body)

    resp = await CLIENT.request(
        'POST', resolve_endpoint(endpoint),
        json=req_body,
        headers=resolve_authorization(authorization),
    )

    if not req_body.get('stream', False):
        try:
            # No streaming. Wait for JSON response and be on our way.
            resp_json = await resp.json()
        finally:
            resp.close()
        return resp_json

    return StreamingResponse(
        LineCopyStreamer(resp), # NB This will close resp once done
        media_type="text/event-stream",
    )


async def simple_proxy(request: Request, endpoint: str, authorization: str|None=None):
    """
    Proxy a simple request (JSON-in -> JSON-out) to another URL.
    """
    assert CLIENT is not None

    req_body = await request.json()
    # pprint(req_body)

    async with CLIENT.request(
        'POST', resolve_endpoint(endpoint),
        json=req_body,
        headers=resolve_authorization(authorization),
    ) as resp:
        return await resp.json()


@app.post('/v1/completions')
async def create_completion(request: Request, authorization: Annotated[str|None, Header()]=None):
    """
    Creates a completion for the provided prompt and parameters.
    """
    return await streaming_aware_proxy(request, '/completions', authorization=authorization)


@app.post('/v1/chat/completions')
async def create_chat_completion(request: Request, authorization: Annotated[str|None, Header()]=None):
    """
    Creates a model response for the given chat conversation.
    """
    return await streaming_aware_proxy(request, '/chat/completions', authorization=authorization)


@app.post('/v1/embeddings')
async def create_embedding(request: Request, authorization: Annotated[str|None, Header()]=None):
    """
    Creates an embedding vector representing the input text.
    """
    return await simple_proxy(request, '/embeddings', authorization=authorization)


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

    if mid in SIMPLE_FILTERS:
        return True

    for rexp in RE_FILTERS:
        if rexp.fullmatch(mid) is not None:
            return True

    return False


class SimpleTTLCache:
    _last_result: Any = None
    _last_time: datetime.datetime|None = None

    async def get(self, func, *args, **kwargs):
        ttl = None if CONFIG.cache_ttl is None else \
            datetime.timedelta(seconds=CONFIG.cache_ttl)

        now = datetime.datetime.now()

        if (ttl is None or
            self._last_time is None or
            (now - self._last_time) > ttl):
            logger.info(f'Cache miss: {func.__name__}')
            self._last_result = await func(*args, **kwargs)
            self._last_time = now
        else:
            logger.info(f'Cache hit: {func.__name__}')

        return self._last_result.copy()


async def _list_models(authorization: str|None) -> ListModelsResponse:
    assert CLIENT is not None

    async with CLIENT.request(
        'GET', resolve_endpoint('/models'),
        headers=resolve_authorization(authorization),
    ) as resp:
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


MODELS_CACHE = SimpleTTLCache()


@app.get('/v1/models', response_model=ListModelsResponse)
async def list_models(authorization: Annotated[str|None, Header()]=None) -> ListModelsResponse:
    """
    Lists the currently available models, and provides basic information about each one such as the owner and availability.
    """
    return await MODELS_CACHE.get(_list_models, authorization)

# TODO Do typical frontends use the other model endpoints?


def main():
    config_file = os.getenv('LLM_MF_CONFIG', DEFAULT_CONFIG)
    host = os.getenv('LLM_MF_HOST', '127.0.0.1')
    port = int(os.getenv('LLM_MF_PORT', 8080))

    parser = argparse.ArgumentParser('OpenAI-compatible API model filter proxy')

    parser.add_argument(
        '-c', '--config',
        type=str,
        default=config_file,
        help=f'Configuration file to use (default: {config_file})'
    )
    parser.add_argument(
        '-H', '--host',
        type=str,
        default=host,
        help=f'Host interface to listen on (default: {host})'
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=port,
        help=f'Port to listen on (default: {port})'
    )

    args = parser.parse_args()

    # Why are we passing this through environment again?
    os.environ['LLM_MF_CONFIG'] = args.config

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == '__main__':
    main()
