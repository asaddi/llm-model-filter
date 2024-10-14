# model_filter.py

An OpenAI-compatible API proxy that filters models.

This is a simple Python server that solves one very specific problem.

Sometimes you might use an OpenAI-like API provider that serves dozens, if not a hundred different models. When you point your frontend at that provider, you're forced to wade through all those different models, even if you only use a handful!

This proxy server will filter the models returned from the `/v1/models` endpoint, either through exact matching (i.e. an explicit allow list) or regexp matching.

All other endpoints are proxied to the real server.

An alternative is to use something like [LiteLLM](https://github.com/BerriAI/litellm) to map "virtual models" (for lack of a better term) to "real" models located at potentially different providers. This is the solution I use today, but I... am a lazy person.

## Features

WIP

## Installation

WIP

## License

Licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
