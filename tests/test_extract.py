import logging
import os
from pathlib import Path
from rich.console import Console
from langchain_core.documents import Document
from rich.panel import Panel
from rich.progress import Progress
from aisuite import Client as AISuiteClient
from llama_api_client import LlamaAPIClient
from pydantic import BaseModel, Field


from lapidarist.read import retrieve_documents
from lapidarist.extract import raw_extraction_template
from lapidarist.extract import partial_formatter
from lapidarist.document_enricher import enrich_document
from lapidarist.document_enricher import make_extract_from_document_chunks

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("lapidarist").setLevel(logging.INFO)

console = Console()

hf_dataset_ids = ["stanfordnlp/imdb"]
hf_dataset_column = "text"
docs_per_dataset = 5
json_enrichment_file = "test-enrichments.json"

aisuite_client = AISuiteClient(
    provider_configs={
        "ollama": {
            "timeout": 180,
        },
        "together": {
            "timeout": 180,
        },
        "openai": {  # Use OpenAI protocol for Llama API access
            "api_key": os.environ.get("LLAMA_API_KEY"),
            "base_url": "https://api.llama.com/compat/v1/",
        },
    }
)

llama_api_client = LlamaAPIClient(
    api_key=os.environ.get("LLAMA_API_KEY"),
    base_url="https://api.llama.com/v1/",
)

# chat_completion_client = llama_api_client
chat_completion_client = aisuite_client

# extraction_model_id = "openai:Llama-4-Maverick-17B-128E-Instruct-FP8"
extraction_model_id = "together:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# extraction_model_id = "llama:Llama-4-Maverick-17B-128E-Instruct-FP8"


def doc_as_rich(doc: Document) -> Panel:
    return Panel(
        doc.page_content,
        title="Document",
    )


class TestDocumentChunkExtractions(BaseModel):
    """
    The geographic locations mentioned in a chunk of a document.
    """

    __test__ = False

    geographic_locations: list[str] = Field(
        description="A list of the geographic locations in the text. For example: ['New Hampshire', 'Portland, Maine', 'Elm Street']"
    )

    movie_titles: list[str] = Field(
        description="A list of the movie titles mentioned in the text. For example: ['Jaws', 'Fletch Lives', 'Rocky IV']"
    )


chunk_extraction_template = partial_formatter.format(
    raw_extraction_template, extraction_description=TestDocumentChunkExtractions.__doc__
)


class TestDocumentEnrichments(BaseModel):
    """
    Enrichments for a document.
    """

    __test__ = False

    # Fields that come directly from the document metadata
    label: int = Field(description="document label")

    # Extracted from the text with LLM
    georefs: list[str] = Field(
        description="A list of the geographic locations mentioned in the text. For example: ['New Hampshire', 'Portland, Maine', 'Elm Street']"
    )

    movierefs: list[str] = Field(
        description="A list of the movie titles mentioned in the text. For example: ['Jaws', 'Fletch Lives', 'Rocky IV']"
    )

    # Written by Lapidarist
    hf_dataset_id: str = Field(description="id of the dataset in HF")
    hf_dataset_index: int = Field(description="index of the document in the HF dataset")


def doc_enrichments(
    doc: Document, chunk_extracts: list[TestDocumentChunkExtractions]
) -> TestDocumentEnrichments:

    # merge information from all chunks
    georefs = []
    movierefs = []
    for chunk_extract in chunk_extracts:
        if chunk_extract.__dict__.get("geographic_locations") is not None:
            georefs.extend(chunk_extract.geographic_locations)
        if chunk_extract.__dict__.get("movie_titles") is not None:
            movierefs.extend(chunk_extract.movie_titles)

    logging.info(doc.metadata)

    enrichments = TestDocumentEnrichments(
        label=doc.metadata["label"],
        georefs=georefs,
        movierefs=movierefs,
        hf_dataset_id=doc.metadata["hf_dataset_id"],
        hf_dataset_index=int(doc.metadata["hf_dataset_index"]),
    )

    return enrichments


def test_retrieve():

    docs = retrieve_documents(
        hf_dataset_ids=hf_dataset_ids,
        hf_dataset_column="text",
        docs_per_dataset=docs_per_dataset,
    )

    assert (
        len(docs) == docs_per_dataset
    ), f"Expected to retrieve {docs_per_dataset} documents from the dataset."


def test_enrich():

    docs = retrieve_documents(
        hf_dataset_ids=hf_dataset_ids,
        hf_dataset_column="text",
        docs_per_dataset=docs_per_dataset,
    )

    extract_from_doc_chunks = make_extract_from_document_chunks(
        doc_as_rich,
        chat_completion_client,
        extraction_model_id,
        chunk_extraction_template,
        TestDocumentChunkExtractions,
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
