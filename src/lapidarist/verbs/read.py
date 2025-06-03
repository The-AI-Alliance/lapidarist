from typing import List
from typing import Optional

import os
import logging

import httpx
from pydantic.networks import HttpUrl
from pathlib import Path

from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.hugging_face_dataset import (
    HuggingFaceDatasetLoader,
)

log = logging.getLogger(__name__)
logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_file(filename: str) -> List[Document]:

    loader = TextLoader(filename)
    documents = loader.load()

    return documents


def load_hugging_face_dataset(
    dataset_name: str, page_content_column: str = "text"
) -> List[Document]:

    loader = HuggingFaceDatasetLoader(
        dataset_name, page_content_column=page_content_column
    )
    documents = loader.load()

    return documents


async def url_to_file(url: HttpUrl, data_file: Path, overwrite: bool = False):

    if data_file.exists() and not overwrite:
        return

    async with httpx.AsyncClient() as client:

        response = await client.get(url)
        response.raise_for_status()

        with open(data_file, "wb") as file:
            file.write(response.content)


def retrieve_documents(
    hf_dataset_ids: list[str], hf_dataset_column: str, docs_per_dataset: int = None
) -> List[Document]:

    docs = []

    for hf_dataset_id in hf_dataset_ids:

        dataset_docs = load_hugging_face_dataset(
            hf_dataset_id, page_content_column=hf_dataset_column
        )

        docs_in_dataset = len(dataset_docs)

        num_docs_to_use = docs_in_dataset
        if docs_per_dataset is not None:
            num_docs_to_use = min(docs_per_dataset, docs_in_dataset)

        log.info(
            f"using {num_docs_to_use}/{docs_in_dataset} documents from {hf_dataset_id}"
        ),

        for i in range(num_docs_to_use):
            doc = dataset_docs[i]
            doc.metadata["hf_dataset_id"] = hf_dataset_id
            doc.metadata["hf_dataset_index"] = i
            docs.append(doc)

    return docs


def retrieve_document(
    hf_dataset_id: str, hf_dataset_column: str, index: int
) -> Optional[Document]:

    docs = load_hugging_face_dataset(
        hf_dataset_id, page_content_column=hf_dataset_column
    )

    if 0 <= index < len(docs):
        return docs[index]
    else:
        return None
