from typing import Optional
import logging
import json
from string import Formatter
from rich.console import Console
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from pydantic import BaseModel
from aisuite import Client as AISuiteClient

log = logging.getLogger(__name__)


def complete_simple(
    client: AISuiteClient, model_id: str, system_prompt: str, user_prompt: str, **kwargs
) -> str:

    console = kwargs.pop("console", None)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if console is not None:

        kwargs_text = "\n".join([str(k) + ": " + str(v) for k, v in kwargs.items()])

        params_text = Text(
            f"""
model_id: {model_id}
{kwargs_text}
    """
        )

        messages_table = Table(title="Messages", show_lines=True)
        messages_table.add_column("Role", justify="left")
        messages_table.add_column("Content", justify="left")  # style="green"
        for message in messages:
            messages_table.add_row(message["role"], message["content"])

        call_panel = Panel(
            Group(params_text, messages_table), title="complete_simple call"
        )
        console.print(call_panel)

    response_text = ""
    if model_id.startswith("llama:"):
        # Llama API Client expects model_id without prefix
        model_id = model_id.replace("llama:", "")
        response = client.chat.completions.create(
            model=model_id, messages=messages, **kwargs
        )
        response_text = response.completion_message.content.text
    else:
        response = client.chat.completions.create(
            model=model_id, messages=messages, **kwargs
        )
        response_text = response.choices[0].message.content

    #    response = client.chat.completions.create(
    #        model=model_id, messages=messages, **kwargs
    #    )
    #    response_text = response.choices[0].message.content

    if console is not None:
        console.print(Panel(response_text, title="Response"))

    return response_text


extraction_system_prompt = "You are an entity extractor"


class PartialFormatter(Formatter):
    def get_value(self, key, args, kwargs):
        try:
            return super().get_value(key, args, kwargs)
        except KeyError:
            return "{" + key + "}"


partial_formatter = PartialFormatter()

raw_extraction_template = """\
Below is a description of a data class for storing information extracted from text:

{extraction_description}

Find the information in the following text, and provide them in the specified JSON response format.
Only answer in JSON.:

{text}
"""


def extract_to_pydantic_model(
    aisuite_client: AISuiteClient,
    extraction_model_id: str,
    extraction_template: str,
    clazz: type[BaseModel],
    text: str,
    console: Optional[Console] = None,
) -> BaseModel:

    extract_str = complete_simple(
        aisuite_client,
        extraction_model_id,
        extraction_system_prompt,
        extraction_template.format(text=text),
        response_format={
            "type": "json_schema",
            "strict": True,
            "schema": clazz.model_json_schema(),
        },
        console=console,
    )

    log.info("complete_to_pydantic_model: extract_str = <<<%s>>>", extract_str)

    try:
        extract_dict = json.loads(extract_str)
        return clazz.model_construct(**extract_dict)
    except Exception as e:
        log.error("complete_to_pydantic_model: Exception: %s", e)

    return None
