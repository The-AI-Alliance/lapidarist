import logging
from pathlib import Path
from rich.progress import Progress
from rich.console import Console
from neo4j import GraphDatabase
import os
from aisuite import Client as AISuiteClient
from llama_api_client import LlamaAPIClient
import pytest

# from neo4j_graphrag.schema import get_schema

from lapidarist.read import retrieve_documents
from lapidarist.document_enricher import enrich_document
from lapidarist.document_enricher import make_extract_from_document_chunks
from lapidarist.entity_resolver import load_entity_resolver
from lapidarist.knowledge_graph import load_knowledge_graph

from movie_demo import (
    hf_dataset_ids,
    hf_dataset_column,
    doc_as_rich,
    chunk_extraction_template,
    ReviewChunkExtractions,
    ReviewEnrichments,
    doc_enrichments,
    doc_enrichments_to_graph,
    resolvers,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("lapidarist").setLevel(logging.INFO)
console = Console()

aisuite_client = AISuiteClient(
    provider_configs={
        "ollama": {
            "timeout": 180,
        },
        "together": {
            "timeout": 180,
        },
        # "openai": {  # Use OpenAI protocol for Llama API access
        #    "api_key": os.environ.get("LLAMA_API_KEY"),
        #    "base_url": "https://api.llama.com/compat/v1/",
        # },
    }
)

llama_api_client = LlamaAPIClient(
    api_key=os.environ.get("LLAMA_API_KEY"),
    base_url="https://api.llama.com/v1/",
)

# chat_completion_client = llama_api_client
chat_completion_client = aisuite_client

docs_per_dataset = 5

# extraction_model_id = "openai:Llama-4-Maverick-17B-128E-Instruct-FP8"
extraction_model_id = "together:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# extraction_model_id = "llama:Llama-4-Maverick-17B-128E-Instruct-FP8"

json_enrichment_file = "test-enrichments.json"

embedding_model_id = "all-MiniLM-L6-v2"
milvus_uri = "file:/bartlebot-milvus.db"

neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_password = os.environ.get("NEO4J_PASSWORD")
neo4j_username = os.environ.get("NEO4J_USERNAME")
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))


@pytest.mark.dependency()
def test_retrieve_and_enrich():

    docs = retrieve_documents(
        hf_dataset_ids=hf_dataset_ids,
        hf_dataset_column=hf_dataset_column,
        docs_per_dataset=docs_per_dataset,
    )

    extract_from_doc_chunks = make_extract_from_document_chunks(
        doc_as_rich,
        chat_completion_client,
        extraction_model_id,
        chunk_extraction_template,
        ReviewChunkExtractions,
        delay=10.0,
        console=console,
    )

    with Progress() as progress:

        task_enrich = progress.add_task(
            "[green]Enriching documents...", total=len(docs)
        )

        with open(json_enrichment_file, "wt") as f:

            for doc in docs:

                enrichments_json = enrich_document(
                    doc, extract_from_doc_chunks, doc_enrichments
                )
                f.write(enrichments_json + "\n")

                progress.update(task_enrich, advance=1)

        log.info("Wrote document enrichments to %s", json_enrichment_file)

    assert Path(
        json_enrichment_file
    ).exists(), f"Expected the enrichment file {json_enrichment_file} to be created."


@pytest.mark.dependency(depends=["test_retrieve_and_enrich"])
def test_load_knowledge_graph():

    load_knowledge_graph(
        neo4j_driver,
        json_enrichment_file,
        ReviewEnrichments,
        doc_enrichments_to_graph,
    )

    assert True, "Knowledge graph loaded successfully."


@pytest.mark.dependency(depends=["test_load_knowledge_graph"])
def test_load_entity_resolver():

    load_entity_resolver(
        neo4j_driver,
        resolvers,
        embedding_model_id,
        milvus_uri,
        console=console,
    )

    assert True, "Entity resolver loaded successfully."
