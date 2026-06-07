"""Knowledge graph construction from spreadsheet rows and document metadata."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
import pandas as pd

from graph_rag.models import DocumentChunk
from graph_rag.utils import normalize_text


class GraphRelationshipBuilder:
    """Extract entities and build typed weighted relationships."""

    def __init__(self) -> None:
        self.entity_map: dict[str, dict[str, Any]] = {}

    def extract_entities_from_excel(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Extract row-level entities from spreadsheet data."""
        entities: list[dict[str, Any]] = []
        mapping = self._dynamic_field_mapping(df.columns)
        for idx, row in df.iterrows():
            for col, value in row.items():
                if col == "_source_file" or pd.isna(value) or not str(value).strip():
                    continue
                entity_type = mapping.get(col, str(col).lower().replace(" ", "_"))
                entity_id = f"excel_{entity_type}_{idx}_{normalize_text(value).replace(' ', '_')[:80]}"
                entity = {
                    "id": entity_id,
                    "name": str(value),
                    "type": entity_type,
                    "source": "excel",
                    "row_index": int(idx),
                    "source_file": row.get("_source_file", "excel"),
                    "attributes": {"row_index": int(idx), "source_file": row.get("_source_file", "excel"), "column": col},
                }
                entities.append(entity)
                self.entity_map[entity_id] = entity
        return entities

    def extract_entities_from_documents(self, chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
        """Extract entities from structured document fields."""
        entities: list[dict[str, Any]] = []
        type_map = {
            "document_number": "document_number",
            "supplier_name": "supplier_name",
            "customer_name": "customer_name",
            "date": "date",
            "amount": "amount",
            "tax_amount": "tax",
            "address": "address",
        }
        seen: set[str] = set()
        for chunk in chunks:
            for field, value in chunk.structured_data.items():
                if not value:
                    continue
                entity_type = type_map.get(field, field)
                entity_id = f"doc_{entity_type}_{normalize_text(value).replace(' ', '_')[:100]}_{chunk.source}"
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                entity = {
                    "id": entity_id,
                    "name": str(value),
                    "type": entity_type,
                    "source": "document",
                    "source_file": chunk.source,
                    "document_type": chunk.document_type,
                    "attributes": {"source_file": chunk.source, "document_type": chunk.document_type, **chunk.structured_data},
                }
                entities.append(entity)
                self.entity_map[entity_id] = entity
        return entities

    def build_relationships(self, excel_entities: list[dict[str, Any]], document_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build cross-source and within-document relationships."""
        relationships: list[dict[str, Any]] = []
        doc_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for entity in document_entities:
            doc_index[(entity["type"], normalize_text(entity["name"]).lower())].append(entity)
        all_types = {e["type"] for e in excel_entities + document_entities}
        for excel_entity in excel_entities:
            normalized_name = normalize_text(excel_entity["name"]).lower()
            for doc_type in all_types:
                if not set(excel_entity["type"].split("_")).intersection(doc_type.split("_")):
                    continue
                for doc_entity in doc_index.get((doc_type, normalized_name), []):
                    relationships.append(self._relationship(excel_entity, doc_entity, self._rel_type(excel_entity["type"], doc_entity["type"]), 1.0))
        relationships.extend(self._star_relationships(excel_entities, "row_index", "SAME_ROW"))
        relationships.extend(self._star_relationships(document_entities, "source_file", "SAME_DOCUMENT"))
        return relationships

    def create_network_graph(self, entities: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> nx.Graph:
        """Create a NetworkX graph from entities and relationships."""
        graph = nx.Graph()
        for entity in entities:
            graph.add_node(entity["id"], name=entity["name"], type=entity["type"], source=entity["source"], **entity.get("attributes", {}))
        for rel in relationships:
            graph.add_edge(rel["from"], rel["to"], type=rel["type"], strength=rel["strength"], description=rel["description"])
        return graph

    def _dynamic_field_mapping(self, columns: Any) -> dict[str, str]:
        patterns = {
            "grn_code": ["grn", "goods receipt"],
            "po_number": ["purchase order", " po", "po no"],
            "invoice_number": ["invoice", "inv"],
            "supplier_name": ["supplier", "vendor"],
            "customer_name": ["customer", "buyer", "client"],
            "address": ["address", "location"],
            "amount": ["amount", "price", "cost", "total"],
            "quantity": ["quantity", "qty", "count"],
            "date": ["date", "created", "updated"],
            "status": ["status", "state"],
        }
        mapping: dict[str, str] = {}
        for col in columns:
            lower = str(col).lower().strip()
            mapping[col] = next((etype for etype, keys in patterns.items() if any(key in lower for key in keys)), lower.replace(" ", "_"))
        return mapping

    def _relationship(self, left: dict[str, Any], right: dict[str, Any], rel_type: str, strength: float) -> dict[str, Any]:
        return {
            "from": left["id"],
            "to": right["id"],
            "type": rel_type,
            "strength": strength,
            "description": f"{left['name']} related to {right['name']}",
        }

    def _rel_type(self, left_type: str, right_type: str) -> str:
        if left_type == right_type:
            return f"SAME_{left_type.upper()}"
        return f"RELATED_{left_type.upper()}_{right_type.upper()}"

    def _star_relationships(self, entities: list[dict[str, Any]], key: str, rel_type: str) -> list[dict[str, Any]]:
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for entity in entities:
            groups[entity.get(key, "unknown")].append(entity)
        relationships: list[dict[str, Any]] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            hub = next((e for e in group if e.get("type") == "document_number"), group[0])
            for entity in group:
                if entity["id"] != hub["id"]:
                    relationships.append(self._relationship(hub, entity, rel_type, 1.0))
        return relationships
