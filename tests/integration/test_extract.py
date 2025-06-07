import logging
from pathlib import Path
from rich.progress import Progress
from rich.console import Console
from neo4j import GraphDatabase
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
    docs_per_dataset,
    doc_as_rich,
    json_enrichment_file,
    chat_completion_client,
    extraction_model_id,
    embedding_model_id,
    milvus_uri,
    neo4j_uri,
    neo4j_username,
    neo4j_password,
    ReviewChunkExtractions,
    ReviewEnrichments,
    resolvers,
    doc_enrichments,
    chunk_extraction_template,
    doc_enrichments_to_graph,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("lapidarist").setLevel(logging.INFO)
console = Console()

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
