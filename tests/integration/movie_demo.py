import logging
from langchain_core.documents import Document
from rich.panel import Panel
from enum import StrEnum
from pydantic import BaseModel, Field

from neomodel import (
    StructuredNode,
    StringProperty,
    IntegerProperty,
    UniqueIdProperty,
    RelationshipTo,
    RelationshipFrom,
    ZeroOrOne,
    ZeroOrMore,
)

from lapidarist.extract import raw_extraction_template
from lapidarist.extract import partial_formatter
from lapidarist.entity_resolver import Resolver
from lapidarist.knowledge_graph import RelationLabel as lapidarist_RelationLabel
from lapidarist.knowledge_graph import Reference

hf_dataset_ids = ["stanfordnlp/imdb"]
hf_dataset_column = "text"


def doc_as_rich(doc: Document) -> Panel:
    return Panel(
        doc.page_content,
        title="Document",
    )


class ReviewChunkExtractions(BaseModel):
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
    raw_extraction_template, extraction_description=ReviewChunkExtractions.__doc__
)


class ReviewEnrichments(BaseModel):
    """
    Enrichments for a document.
    """

    __test__ = False

    # Fields that come directly from the document metadata
    label: int = Field(description="document label")
    text: str = Field(description="The text of the movie review.")

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
    doc: Document, chunk_extracts: list[ReviewChunkExtractions]
) -> ReviewEnrichments:

    # merge information from all chunks
    georefs = []
    movierefs = []
    for chunk_extract in chunk_extracts:
        if chunk_extract.__dict__.get("geographic_locations") is not None:
            georefs.extend(chunk_extract.geographic_locations)
        if chunk_extract.__dict__.get("movie_titles") is not None:
            movierefs.extend(chunk_extract.movie_titles)

    logging.info(doc.metadata)

    enrichments = ReviewEnrichments(
        label=doc.metadata["label"],
        text=doc.page_content,
        georefs=georefs,
        movierefs=movierefs,
        hf_dataset_id=doc.metadata["hf_dataset_id"],
        hf_dataset_index=int(doc.metadata["hf_dataset_index"]),
    )

    return enrichments


class NodeLabel(StrEnum):
    REVIEW = "Review"
    MOVIE = "Movie"
    MOVIE_REFERENCE = "MovieReference"
    GEO = "Geo"
    GEO_REFERENCE = "GeoReference"


class Review(StructuredNode):
    uid = UniqueIdProperty()
    text = StringProperty(required=True)

    movie_mentions = RelationshipTo(
        NodeLabel.MOVIE_REFERENCE,
        lapidarist_RelationLabel.MENTIONS,
        cardinality=ZeroOrMore,
    )
    geo_mentions = RelationshipTo(
        NodeLabel.GEO_REFERENCE,
        lapidarist_RelationLabel.MENTIONS,
        cardinality=ZeroOrMore,
    )

    hf_dataset_id = StringProperty()
    hf_dataset_index = IntegerProperty()


class Movie(StructuredNode):
    uid = UniqueIdProperty()
    title = StringProperty(required=True)


class MovieReference(Reference):
    referent = RelationshipTo(
        Movie, lapidarist_RelationLabel.REFERS_TO, cardinality=ZeroOrOne
    )
    referred_to_by = RelationshipFrom(
        NodeLabel.MOVIE_REFERENCE,
        lapidarist_RelationLabel.REFERS_TO,
        cardinality=ZeroOrMore,
    )


class Geo(StructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(required=True)
    referred_to_by = RelationshipFrom(
        NodeLabel.GEO_REFERENCE,
        lapidarist_RelationLabel.REFERS_TO,
        cardinality=ZeroOrMore,
    )


class GeoReference(Reference):
    referent = RelationshipTo(
        Geo, lapidarist_RelationLabel.REFERS_TO, cardinality=ZeroOrOne
    )
    referred_to_by = RelationshipFrom(
        NodeLabel.GEO_REFERENCE,
        lapidarist_RelationLabel.REFERS_TO,
        cardinality=ZeroOrMore,
    )


def doc_enrichments_to_graph(enrichments: ReviewEnrichments) -> None:

    review_node = Review(
        label=enrichments.label,
        text=enrichments.text,
        hf_dataset_id=enrichments.hf_dataset_id,
        hf_dataset_index=enrichments.hf_dataset_index,
    ).save()

    for movieref in enrichments.movierefs:
        review_node.movie_mentions.connect(MovieReference(text=movieref).save())

    for georef in enrichments.georefs:
        review_node.geo_mentions.connect(GeoReference(text=georef).save())


movie_resolver = Resolver(
    "MATCH (mr:MovieRef) RETURN mr.text AS text",
    "text",
    "resolve_movierefs",
)

geo_resolver = Resolver(
    "MATCH (gr:GeoRef) RETURN gr.text AS text",
    "text",
    "resolve_georefs",
)

resolvers = [movie_resolver, geo_resolver]
