from typing import List
from typing import Callable
from typing import Optional

import time
import logging
from aisuite import Client as AISuiteClient
from pydantic import BaseModel

from rich.panel import Panel
from rich.console import Console

from langchain_core.documents.base import Document

from lapidarist.chunk import documents_to_chunks_by_tokens
from lapidarist.extract import extract_to_pydantic_model

log = logging.getLogger(__name__)


def extract_from_document_chunks(
    doc: Document,
    doc_as_rich: Callable[[Document], Panel],
    aisuite_client: AISuiteClient,
    chunk_extraction_model_id: str,
    chunk_extraction_template: str,
    chunk_extract_clazz: type[BaseModel],
    delay: float,
    console: Optional[Console] = None,
) -> List[BaseModel]:

    if console is not None:
        console.print(doc_as_rich(doc))
        console.print()

    extract_models = []

    chunks = documents_to_chunks_by_tokens([doc], chunk_size=1000, chunk_overlap=0)
    for i, chunk in enumerate(chunks):

        ce = extract_to_pydantic_model(
            aisuite_client,
            chunk_extraction_model_id,
            chunk_extraction_template,
            chunk_extract_clazz,
            chunk.page_content,
        )

        log.info("Extract model in chunk %s of %s", i + 1, len(chunks))
        if console is not None:
            console.print(Panel(str(ce)))

        extract_models.append(ce)
        time.sleep(delay)

    return extract_models


def make_extract_from_document_chunks(
    doc_as_rich: Callable[[Document], Panel],
    aisuite_client: AISuiteClient,
    chunk_extraction_model_id: str,
    chunk_extraction_template: str,
    chunk_extract_clazz: type[BaseModel],
    delay: float = 1.0,  # intra-chunk delay between inference calls
    console: Optional[Console] = None,
) -> Callable[[Document, bool], List[BaseModel]]:

    def fn(doc: Document) -> List[BaseModel]:

        chunk_extract_models = extract_from_document_chunks(
            doc,
            doc_as_rich,
            aisuite_client,
            chunk_extraction_model_id,
            chunk_extraction_template,
            chunk_extract_clazz,
            delay,
            console=console,
        )

        return chunk_extract_models

    return fn


def enrich_document(
    doc: Document,
    extract_from_doc_chunks: Callable[[Document], List[BaseModel]],
    doc_enrichments: Callable[[Document, list[BaseModel]], BaseModel],
) -> dict:
    chunk_extract_models = extract_from_doc_chunks(doc)
    enrichments = doc_enrichments(doc, chunk_extract_models)
    enrichments_json = enrichments.model_dump_json()
    return enrichments_json
