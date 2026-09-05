#!/usr/bin/env python3
"""Server-backed rendering/tokenization adapter for the shared prompt fitter."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pager_benchmark_contract import RenderedPrompt, PromptSizingError


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class ServerPromptRenderer:
    """Render and tokenize complete chat messages through one running server."""

    def __init__(self, endpoint: str, model: str, api_key: str = "", *,
                 timeout: float = 10.0, request_options: Mapping[str, Any] | None = None,
                 tokenizer_id: str | None = None) -> None:
        self.endpoint = endpoint.split("/v1/", 1)[0].rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.request_options = dict(request_options or {})
        self.tokenizer_id = tokenizer_id or f"server-model:{model}"
        self.template_id = self._template_identity()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint + path, data=json.dumps(body).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (HTTPError, OSError, URLError, TimeoutError, ValueError) as error:
            raise PromptSizingError(f"server prompt endpoint failed: {type(error).__name__}") from error
        if not isinstance(value, dict):
            raise PromptSizingError("server prompt endpoint returned a non-object")
        return value

    def _get(self, path: str) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with urlopen(Request(self.endpoint + path, headers=headers), timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (HTTPError, OSError, URLError, TimeoutError, ValueError) as error:
            raise PromptSizingError(f"server prompt endpoint failed: {type(error).__name__}") from error
        if not isinstance(value, dict):
            raise PromptSizingError("server prompt endpoint returned a non-object")
        return value

    def _template_identity(self) -> str:
        try:
            props = self._get("/props")
            template = props.get("chat_template")
            if not isinstance(template, str):
                raise PromptSizingError("server props response has no chat template")
            return "server-chat-template-sha256:" + _digest({
                "template": template,
                "tool_template": props.get("chat_template_tool_use"),
                "chat_template_kwargs": self.request_options.get("chat_template_kwargs", {}),
            })
        except PromptSizingError:
            # The renderer remains usable with older deployments that expose
            # apply-template/tokenize but not props; identity is explicit and
            # never mistaken for a local tokenizer identity.
            return "server-apply-template-v1:" + _digest({
                "model": self.model,
                "chat_template_kwargs": self.request_options.get("chat_template_kwargs", {}),
            })

    def __call__(self, messages: list[dict[str, Any]]) -> RenderedPrompt:
        body = dict(self.request_options)
        body.update({"model": self.model, "messages": messages})
        rendered = self._post("/apply-template", body).get("prompt")
        if not isinstance(rendered, str):
            raise PromptSizingError("server apply-template response has no prompt")
        tokenized = self._post("/tokenize", {
            "content": rendered,
            "add_special": True,
            "parse_special": True,
        })
        token_ids = tokenized.get("tokens")
        if not isinstance(token_ids, list) or any(not isinstance(token, int) for token in token_ids):
            raise PromptSizingError("server tokenize response has no integer token list")
        return RenderedPrompt(rendered, tuple(token_ids), self.template_id, self.tokenizer_id)


def request_options(*, chat_template_kwargs: Mapping[str, Any] | None = None,
                    tools: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if chat_template_kwargs is not None:
        options["chat_template_kwargs"] = dict(chat_template_kwargs)
    if tools is not None:
        options["tools"] = [dict(tool) for tool in tools]
    return options
