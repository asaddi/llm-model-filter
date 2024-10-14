# model_filter.py

An OpenAI-compatible API proxy that filters models.

This is a simple Python server that solves one very specific problem.

Sometimes you might use an OpenAI-like API provider that serves dozens, if not a hundred different models. When you point your frontend at that provider, you're forced to wade through all those different models, even if you only use a handful!

This proxy server will filter the models returned from the `/v1/models` endpoint, either through exact matching (i.e. an explicit allow list) or regexp matching.

All other endpoints are proxied to the real server.

An alternative is to use something like [LiteLLM](https://github.com/BerriAI/litellm) to map "virtual models" (for lack of a better term) to "real" models located at potentially different providers. This is the solution I use today, but I... am a lazy person.

## Features

* [x] Regexp and exact-match filtering of `/v1/models` endpoint
* [x] Defaults to case-insensitive matching of both regexp and exact-match filters
* [x] `/v1/chat/completions` and `/v1/completions` endpoints are proxied, supporting streaming and non-streaming requests
* [x] `/v1/embeddings` is proxied
* [x] If present, the `Authorization` header (which contains your API key), is simply passed through to the proxied service
* [x] Simple TTL caching of filtered `/v1/models` output
* [x] Dockerfile
* [ ] Robustness. I know failing authorization simply leaves the client in the dark with a generic 5xx error. Would be better if the proxy's endpoint simply mirrored the HTTP status code/message of the underlying service.

### Future?

* If there's an API endpoint you want to add, please open a PR or issue. It should be easy since we proxy everything but `/v1/models`

* Wouldn't it be cool if it could proxy multiple backends? Assign a model prefix to each backend, maybe assign each backend its own set of filters. It would route based on the prefix of the received `model` parameter.

   Querying `/v1/models` would query **all** backends and return a list that is a union of all models (prefix added) that passed their respective filters.

   Of course this might be challenging with the `Authorization` header pass-through. How to ensure that it *never* sends the wrong header to a service?

## Installation

    pip install -r requirements.txt

or if you prefer unpinned dependencies:

    pip install -r requirements.in

## Running

You can set the following environment variables in lieu of CLI arguments:

* `LLM_MF_CONFIG`: Config file, defaults to `config.yaml` in the current directory
* `LLM_MF_HOST`: Host address to bind to. Defaults to 127.0.0.1
* `LLM_MF_PORT`: Port to listen at. Defaults to 8080.

CLI arguments (all optional) override environment.

    python model_filter.py --config <path/to/config> --host <host> --port <port>

## License

Licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
