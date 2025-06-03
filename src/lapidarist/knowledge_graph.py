from typing import Callable

from enum import StrEnum
import logging
import json
from pydantic import BaseModel
from rich.progress import Progress
from neo4j import Driver
from neomodel import (
    db,
    StructuredNode,
    StringProperty,
    UniqueIdProperty,
    RelationshipFrom,
    One,
)

log = logging.getLogger(__name__)


class RelationLabel(StrEnum):
    MENTIONS = "MENTIONS"
    REFERS_TO = "REFERS_TO"


class Reference(StructuredNode):
    uid = UniqueIdProperty()
    referer = RelationshipFrom(StructuredNode, RelationLabel.MENTIONS, cardinality=One)
    text = StringProperty(required=True)
    # confidence = StringProperty(default="0.0")
    # referent = RelationshipTo(_, RelationLabel.REFERS_TO, cardinality=ZeroOrOne)


def load_knowledge_graph(
    driver: Driver,
    enrichments_jsonl_file: str,
    enrichments_clazz: type[BaseModel],
    doc_enrichments_to_graph: Callable[[BaseModel], None],
) -> None:

    db.set_connection(driver=driver)

    log.info("Parsing enrichments from %s", enrichments_jsonl_file)

    enrichmentss = []
    with open(enrichments_jsonl_file, "r") as f:
        for line in f:
            e = enrichments_clazz.model_construct(**json.loads(line))
            enrichmentss.append(e)

    with Progress() as progress:

        task_load = progress.add_task(
            f"Loading {len(enrichmentss)} enriched documents into graph...",
            total=len(enrichmentss),
        )

        with db.transaction:

            log.info("Clearing the graph")
            db.cypher_query("MATCH (n) DETACH DELETE n")

            log.info("Loading %s enriched documents into the graph", len(enrichmentss))
            for e in enrichmentss:
                doc_enrichments_to_graph(e)
                progress.update(task_load, advance=1)
