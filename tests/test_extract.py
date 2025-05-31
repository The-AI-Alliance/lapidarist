import logging
from rich.console import Console

from lapidarist.verbs.read import retrieve_documents

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("lapidarist").setLevel(logging.INFO)

console = Console()


def test_extract():

    docs = retrieve_documents(
        hf_dataset_ids=["stanfordnlp/imdb"],
        hf_dataset_column="text",
        docs_per_dataset=10,
    )

    assert len(docs) == 10, "Expected to retrieve 10 documents from the dataset."

    assert True, "This is a placeholder test for the extract functionality."

    # production.law_library.doc_enrichments.build()

    # production.law_library.case_law_knowledge_graph.build()

    # production.law_library.entity_resolvers.build()

    # graph rag test
