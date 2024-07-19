import streamlit as st
import inspect
import numpy as np
import json
from mistralai.models.chat_completion import (
    ChatMessage,
    ResponseFormat,
    ResponseFormats,
    Function,
    ToolChoice,
)
from pydantic import BaseModel
from mistralai.client import MistralClient
from dataclasses import fields, is_dataclass
from string import punctuation


DEFAULT_SYSTEM_PROMPT = (
    "You are a polite chatbot who always answers truthfully and concisely."
)
DEFAULT_USER_PROMPT = "{input}"


class Chatbot:
    def __init__(
        self,
        model: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt=DEFAULT_USER_PROMPT,
    ) -> None:
        self.model = model
        self.client = MistralClient(api_key=st.secrets.mistral_api_key)

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        if "history" not in st.session_state:
            st.session_state.history = []

        self.functions = {}

    def tool(self, **parameters):
        types = {str: "string", int: "integer", float: "float"}

        def decorator(fn):
            params = {}

            for argname, arg in inspect.signature(fn).parameters.items():
                if arg.kind in [arg.VAR_KEYWORD, arg.VAR_POSITIONAL]:
                    continue

                params[argname] = dict(
                    type=types[arg.annotation], description=parameters[argname]
                )

            function = Function(
                name=fn.__name__,
                description=fn.__doc__,
                parameters=dict(
                    type="object", properties=params, required=list(params)
                ),
            )

            self.functions[fn.__name__] = dict(fn=fn, function=function)
            return fn

        return decorator

    def reset(self):
        st.session_state.history.clear()

    def store(self, role, content):
        st.session_state.history.append(dict(content=content, role=role))

    def history(self, context="all", parse=True):
        if context == 0:
            return []

        if context == "all":
            messages = st.session_state.history
        else:
            messages = st.session_state.history[-context:]

        if parse:
            return [ChatMessage(**m) for m in messages]
        else:
            return messages

    def _tools(self):
        return [
            dict(type="function", function=f["function"])
            for f in self.functions.values()
        ]

    def _stream(self, messages, force_tools=False):
        result = []

        for chunk in self.client.chat_stream(
            messages=messages, model=self.model
        ):
            text = chunk.choices[0].delta.content
            result.append(text)
            yield text

        self.store("assistant", "".join(result))

    def _chat(self, messages, force_tools=False):
        response = (
            self.client.chat(
                messages=messages,
                model=self.model,
                tools=self._tools(),
                tool_choice=ToolChoice.any if force_tools else ToolChoice.auto,
            )
            .choices[0]
            .message
        )

        print(response, flush=True)

        if response.tool_calls:
            for call in response.tool_calls:
                func_name = call.function.name
                args = json.loads(call.function.arguments)
                result=self.functions[func_name]["fn"](**args)

                if result is None:
                    continue

                result = dict(
                    function=func_name, result=result
                )

                messages.append(response)
                messages.append(ChatMessage(role="tool", content=json.dumps(result)))

            response = (
                self.client.chat(
                    messages=messages,
                    model=self.model,
                    tools=self._tools(),
                    tool_choice=ToolChoice.none,
                )
                .choices[0]
                .message
            )

            print(response, flush=True)

        self.store("assistant", response.content)

        return response.content

    def submit(
        self, content, role="user", context=0, stream=True, force_tools=False, store=True, **kwargs
    ):
        messages = self.history(context)

        print(messages, flush=True)

        if store:
            self.store(role, content)

        messages.insert(0, ChatMessage(role="system", content=self.system_prompt))

        if role == "user":
            content = self.user_prompt.format(input=content, **kwargs)

        messages.append(ChatMessage(role=role, content=content))

        if role == "user":
            if stream:
                return self._stream(messages, force_tools=force_tools)
            else:
                return self._chat(messages, force_tools=force_tools)

    def create(self, model: type[BaseModel], prompt, context=0, retries=3):
        if not issubclass(model, BaseModel):
            raise TypeError("`model` must be a dataclass type.")

        params = {}
        types = {str: "string", int: "integer", float: "float"}

        for field, info in model.model_fields.items():
            params[field] = dict(
                type=types[info.annotation], description=info.description
            )

        function = Function(
            name=f"create_{model.__name__}",
            description=model.__doc__,
            parameters=dict(type="object", properties=params, required=list(params)),
        )

        messages = self.history(context)

        messages.insert(0, ChatMessage(role="system", content=self.system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        response = (
            self.client.chat(
                messages,
                self.model,
                tools=[dict(type="function", function=function)],
                tool_choice=ToolChoice.any,
            )
            .choices[0]
            .message
        )

        try:
            result = json.loads(response.tool_calls[0].function.arguments)
            return model(**result)
        except:
            print(response, flush=True)

            if retries <= 0:
                raise

            return self.create(prompt, model, context, retries - 1)

    def json(self, content, context=0, model=None, retries: int = 3):
        if model and (not isinstance(model, type) or not is_dataclass(model)):
            raise TypeError("`model` must be a dataclass type.")

        messages = self.history(context)

        messages.insert(0, ChatMessage(role="system", content=self.system_prompt))
        messages.append(ChatMessage(role="user", content=content))

        response = (
            self.client.chat(
                messages,
                self.model,
                response_format=ResponseFormat(type=ResponseFormats.json_object),
            )
            .choices[0]
            .message.content
        )

        try:
            result = json.loads(response)

            if model:
                return _parse_dataclass(result, model)

            return result
        except:
            print(response, flush=True)

            if retries <= 0:
                raise

            return self.json(content, context, model, retries - 1)


def _parse_dataclass(json_obj, model):
    """
    Flexible parsing of JSON objects into dataclasses.

    All keys in `json_obj` are tested against the `model` fields
    and the "best match" is selected. If there is an exact match,
    that is used. Otherwise, the first key in `json_obj` that has
    the corresponding field name as substring is selected.

    Args:
        - json_obj: a dict-like object, e.g., the result of `json.loads`.
        - model: a dataclass type.

    Returns: An instance of `model` initialized with the arguments in `json_obj`.

    Examples:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class User:
    ...     name: str
    ...     age: int

    Perfect matches work:

    >>> obj = { "age": 35, "name": "Alex" }
    >>> _parse_dataclass(obj, User)
    User(name='Alex', age=35)

    Wrong case also works:

    >>> obj = { "Age": 35, "Name": "Alex" }
    >>> _parse_dataclass(obj, User)
    User(name='Alex', age=35)

    Extra symbols in the object key:

    >>> obj = { "`age`": 35, "?>name": "Alex" }
    >>> _parse_dataclass(obj, User)
    User(name='Alex', age=35)

    Superfluos keys:

    >>> obj = { "age": 35, "name": "Alex", "extra": False }
    >>> _parse_dataclass(obj, User)
    User(name='Alex', age=35)

    Wrong (but casteable) types:

    >>> obj = { "age": "35", "name": "Alex", "extra": False }
    >>> _parse_dataclass(obj, User)
    User(name='Alex', age=35)

    """
    result = dict()

    for field in fields(model):
        value = json_obj.get(field.name)

        if not value:
            for k, v in json_obj.items():
                if field.name.lower() == k.lower().strip(punctuation):
                    value = v
                    break

        if value:
            result[field.name] = field.type(value)

    return model(**result)


class VectorStore:
    def __init__(self) -> None:
        self._data = []
        self._vectors = []

    def __len__(self):
        return len(self._data)

    def add(self, texts):
        self._data.extend(texts)
        self._vectors.extend(self.embed(texts))

    def embed(self, texts, max_tokens=10000):
        client = MistralClient(api_key=st.secrets.mistral_api_key)
        batch = []
        embeddings = []
        tokens = 0

        def _embed(batch):
            if len(batch) == 0:
                return []

            try:
                response = client.embeddings("mistral-embed", batch)
                return [np.asarray(d.embedding) for d in response.data]
            except Exception as e:
                if not "Too many tokens" in str(e):
                    raise

                if len(batch) == 1:
                    raise

                return _embed(batch[: len(batch) // 2]) + _embed(
                    batch[len(batch) // 2 :]
                )

        for text in texts:
            words = len(text.split())

            if tokens + words >= max_tokens:
                _embed(batch)
                batch = []
                tokens = 0

            batch.append(text)
            tokens += words

        if batch:
            embeddings.extend(_embed(batch))

        return embeddings

    def search(self, text, k=1):
        # Get the document embedding
        v = self.embed([text])[0]
        # Find the closest K vectors
        idx_min = sorted(
            range(len(self._vectors)),
            key=lambda i: np.linalg.norm(v - self._vectors[i]),
        )[:k]
        # Retrieve the corresponding texts
        return [self._data[i] for i in idx_min]
