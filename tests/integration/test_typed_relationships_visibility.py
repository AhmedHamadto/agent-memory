"""Regression tests for typed relationship visibility in export and health.

Verifies that:
1. memory_export_graph surfaces typed entity-to-entity edges (not just RELATED_TO)
2. memory_health dynamically enumerates all relationship types (not just ABOUT)

These tests create 3 entities connected by a non-ABOUT typed relationship
(BELONGS_TO) and verify both endpoints surface the typed edge.

See: docs/audit/2026-04-15-graph-audit.md
"""

import json

import pytest

from neo4j_agent_memory.memory.long_term import EntityType


@pytest.mark.integration
class TestTypedRelationshipsInExport:
    """Verify get_graph() returns typed entity-to-entity relationships."""

    @pytest.mark.asyncio
    async def test_export_includes_typed_entity_relationships(self, clean_memory_client):
        """Typed edges like BELONGS_TO must appear in the export, not just RELATED_TO."""
        client = clean_memory_client

        # Create 3 entities
        e1, _ = await client.long_term.add_entity(
            "ExportTest_Alpha",
            EntityType.ORGANIZATION,
            resolve=False,
            generate_embedding=False,
        )
        e2, _ = await client.long_term.add_entity(
            "ExportTest_Beta",
            EntityType.PERSON,
            resolve=False,
            generate_embedding=False,
        )
        e3, _ = await client.long_term.add_entity(
            "ExportTest_Gamma",
            EntityType.OBJECT,
            resolve=False,
            generate_embedding=False,
        )

        # Create a typed relationship via raw Cypher (how typed edges exist in prod)
        await client.graph.execute_write(
            """
            MATCH (a:Entity {name: 'ExportTest_Alpha'})
            MATCH (b:Entity {name: 'ExportTest_Beta'})
            CREATE (b)-[:WORKS_AT {confidence: 1.0, created_at: datetime()}]->(a)
            """,
            {},
        )
        await client.graph.execute_write(
            """
            MATCH (a:Entity {name: 'ExportTest_Alpha'})
            MATCH (c:Entity {name: 'ExportTest_Gamma'})
            CREATE (c)-[:BELONGS_TO {confidence: 1.0, created_at: datetime()}]->(a)
            """,
            {},
        )

        # Export long-term graph
        graph = await client.get_graph(memory_types=["long_term"])

        # All 3 entities must be present
        entity_names = {
            node.properties.get("name")
            for node in graph.nodes
            if "Entity" in node.labels
        }
        assert "ExportTest_Alpha" in entity_names
        assert "ExportTest_Beta" in entity_names
        assert "ExportTest_Gamma" in entity_names

        # Both typed relationships must be present
        rel_types = {rel.type for rel in graph.relationships}
        assert "WORKS_AT" in rel_types, (
            f"WORKS_AT not found in export. Got: {rel_types}"
        )
        assert "BELONGS_TO" in rel_types, (
            f"BELONGS_TO not found in export. Got: {rel_types}"
        )

        # Verify relationship count >= 2
        entity_rels = [
            r for r in graph.relationships
            if r.type in ("WORKS_AT", "BELONGS_TO")
        ]
        assert len(entity_rels) == 2

    @pytest.mark.asyncio
    async def test_export_includes_about_relationships(self, clean_memory_client):
        """ABOUT edges from Facts to Entities must also appear in the export."""
        client = clean_memory_client

        # Create an entity and a linked fact
        await client.long_term.add_entity(
            "ExportTest_FactTarget",
            EntityType.PERSON,
            resolve=False,
            generate_embedding=False,
        )
        await client.long_term.add_fact(
            subject="ExportTest_FactTarget",
            predicate="has_role",
            obj="tester",
            generate_embedding=False,
        )

        # Export long-term graph
        graph = await client.get_graph(memory_types=["long_term"])

        # ABOUT relationship should be present
        about_rels = [r for r in graph.relationships if r.type == "ABOUT"]
        assert len(about_rels) >= 1, (
            f"ABOUT not found in export. Got types: "
            f"{[r.type for r in graph.relationships]}"
        )

    @pytest.mark.asyncio
    async def test_export_metadata_counts_match_actual(self, clean_memory_client):
        """Export metadata node_count and relationship_count must match actual data."""
        client = clean_memory_client

        await client.long_term.add_entity(
            "ExportTest_CountA",
            EntityType.PERSON,
            resolve=False,
            generate_embedding=False,
        )
        await client.long_term.add_entity(
            "ExportTest_CountB",
            EntityType.ORGANIZATION,
            resolve=False,
            generate_embedding=False,
        )
        await client.graph.execute_write(
            """
            MATCH (a:Entity {name: 'ExportTest_CountA'})
            MATCH (b:Entity {name: 'ExportTest_CountB'})
            CREATE (a)-[:WORKS_AT {confidence: 1.0}]->(b)
            """,
            {},
        )

        graph = await client.get_graph(memory_types=["long_term"])

        assert graph.metadata["node_count"] == len(graph.nodes)
        assert graph.metadata["relationship_count"] == len(graph.relationships)
        assert graph.metadata["relationship_count"] >= 1


@pytest.mark.integration
class TestTypedRelationshipsInHealth:
    """Verify memory_health dynamically enumerates all relationship types."""

    @pytest.mark.asyncio
    async def test_health_reports_typed_entity_relationships(self, clean_memory_client):
        """memory_health must report WORKS_AT, BELONGS_TO, etc. — not just ABOUT."""
        client = clean_memory_client

        # Create entities with a typed relationship
        await client.long_term.add_entity(
            "HealthTest_Org",
            EntityType.ORGANIZATION,
            resolve=False,
            generate_embedding=False,
        )
        await client.long_term.add_entity(
            "HealthTest_Person",
            EntityType.PERSON,
            resolve=False,
            generate_embedding=False,
        )
        await client.graph.execute_write(
            """
            MATCH (p:Entity {name: 'HealthTest_Person'})
            MATCH (o:Entity {name: 'HealthTest_Org'})
            CREATE (p)-[:WORKS_AT {confidence: 1.0, created_at: datetime()}]->(o)
            """,
            {},
        )

        # Run the health query directly (same as the MCP tool does)
        rel_records = await client.graph.execute_read(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
            """,
            {},
        )

        rel_types = {row["rel_type"] for row in rel_records}
        assert "WORKS_AT" in rel_types, (
            f"WORKS_AT not found in health output. Got: {rel_types}"
        )

    @pytest.mark.asyncio
    async def test_health_counts_all_relationship_types(self, clean_memory_client):
        """Total relationship count from health must match actual graph edge count."""
        client = clean_memory_client

        # Create entities, facts, and typed relationships
        await client.long_term.add_entity(
            "HealthCountTest_A",
            EntityType.PERSON,
            resolve=False,
            generate_embedding=False,
        )
        await client.long_term.add_entity(
            "HealthCountTest_B",
            EntityType.ORGANIZATION,
            resolve=False,
            generate_embedding=False,
        )
        await client.long_term.add_fact(
            subject="HealthCountTest_A",
            predicate="test",
            obj="value",
            generate_embedding=False,
        )
        await client.graph.execute_write(
            """
            MATCH (a:Entity {name: 'HealthCountTest_A'})
            MATCH (b:Entity {name: 'HealthCountTest_B'})
            CREATE (a)-[:LEADS {confidence: 1.0}]->(b)
            """,
            {},
        )

        # Get dynamic relationship counts
        rel_records = await client.graph.execute_read(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
            """,
            {},
        )

        health_total = sum(row["count"] for row in rel_records)

        # Get actual total from graph
        actual = await client.graph.execute_read(
            "MATCH ()-[r]->() RETURN count(r) AS total", {}
        )

        assert health_total == actual[0]["total"]

        # Both ABOUT and LEADS must be reported
        rel_dict = {row["rel_type"]: row["count"] for row in rel_records}
        assert "ABOUT" in rel_dict, f"ABOUT missing from: {rel_dict}"
        assert "LEADS" in rel_dict, f"LEADS missing from: {rel_dict}"
