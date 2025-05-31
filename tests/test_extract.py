import logging
from rich.console import Console
from langchain_core.documents import Document
from rich.panel import Panel
from pydantic import BaseModel, Field

from lapidarist.verbs.read import retrieve_documents
from lapidarist.verbs.read import retriever
from lapidarist.verbs.extract import raw_extraction_template
from lapidarist.verbs.extract import partial_formatter
from lapidarist.patterns.document_enricher import enrich_documents
from lapidarist.patterns.document_enricher import make_extract_from_document_chunks

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("lapidarist").setLevel(logging.INFO)

console = Console()

hf_dataset_ids = ["stanfordnlp/imdb"]
hf_dataset_column = "text"
docs_per_dataset = 10
json_enrichment_file = "test-enrichments.json"

extraction_model_id = "together:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"


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


chunk_extraction_template = partial_formatter.format(
    raw_extraction_template, extraction_description=TestDocumentChunkExtractions.__doc__
)


class TestDocumentEnrichments(BaseModel):
    """
    Enrichments for a document.
    """

    __test__ = False

    # Fields that come directly from the document metadata
    label: str = Field(description="document label")

    # Extracted from the text with LLM
    georefs: list[str] = Field(
        description="A list of the geographic locations mentioned in the text. For example: ['New Hampshire', 'Portland, Maine', 'Elm Street']"
    )

    # Written by Lapidarist
    hf_dataset_id: str = Field(description="id of the dataset in HF")
    hf_dataset_index: int = Field(description="index of the document in the HF dataset")


def doc_enrichments(
    doc: Document, chunk_extracts: list[TestDocumentChunkExtractions]
) -> TestDocumentEnrichments:

    # merge information from all chunks
    georefs = []
    for chunk_extract in chunk_extracts:
        if chunk_extract.__dict__.get("geographic_locations") is not None:
            georefs.extend(chunk_extract.geographic_locations)

    logging.info(doc.metadata)

    enrichments = TestDocumentEnrichments(
        label=doc.metadata["label"],
        georefs=georefs,
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

    extract_from_doc_chunks = make_extract_from_document_chunks(
        doc_as_rich,
        extraction_model_id,
        chunk_extraction_template,
        TestDocumentChunkExtractions,
        delay=2.0,
        console=console,
    )

    enrich_documents(
        retriever(hf_dataset_ids, hf_dataset_column, docs_per_dataset),
        extract_from_doc_chunks,
        doc_enrichments,
        json_enrichment_file,
        console=console,
    )
