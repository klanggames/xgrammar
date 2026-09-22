"""Render Gemma-4 prompts from the vendored chat template.

Gemma-4's Hugging Face repos are gated, so the alignment test cannot load the
tokenizer. ``chat_template_gemma4.jinja`` in this directory is a copy of vLLM's
``examples/tool_chat_template_gemma4.jinja``, rendered here through the same
jinja2 environment transformers uses in ``_compile_jinja_template`` so the
result matches ``tokenizer.apply_chat_template``.
"""

import os
from functools import lru_cache

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), "chat_template_gemma4.jinja")

bos_token = "<bos>"


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


@lru_cache(maxsize=1)
def _load_template():
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.globals["raise_exception"] = _raise_exception
    with open(TEMPLATE_FILE, encoding="utf-8") as f:
        return env.from_string(f.read())


def encode_messages(messages, tools=None, enable_thinking=False, add_generation_prompt=False):
    return _load_template().render(
        messages=messages,
        tools=tools,
        enable_thinking=enable_thinking,
        add_generation_prompt=add_generation_prompt,
        bos_token=bos_token,
    )
