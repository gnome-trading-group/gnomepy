import pytest

from gnomepy.registry.graph import RelationshipGraph
from gnomepy.registry.types import ContractRelationship


def make_rel(rel_id, id_a, id_b, rel_type, confidence=0.95, method="semantic"):
    return ContractRelationship(
        relationship_id=rel_id,
        security_id_a=id_a,
        security_id_b=id_b,
        relationship_type=rel_type,
        confidence=confidence,
        method=method,
        date_created="2025-01-01T00:00:00",
    )


class TestRelationshipGraphEmpty:
    def test_get_equivalents_empty(self):
        g = RelationshipGraph([])
        assert g.get_equivalents(1) == []

    def test_get_complement_empty(self):
        g = RelationshipGraph([])
        assert g.get_complement(1) == []

    def test_get_equivalent_pairs_empty(self):
        g = RelationshipGraph([])
        assert g.get_equivalent_pairs() == []

    def test_get_related_empty(self):
        g = RelationshipGraph([])
        assert g.get_related(1, "EQUIVALENT") == []


class TestRelationshipGraphEquivalents:
    def setup_method(self):
        self.g = RelationshipGraph([make_rel(1, 100, 200, "EQUIVALENT")])

    def test_forward(self):
        assert self.g.get_equivalents(100) == [200]

    def test_reverse_symmetric(self):
        assert self.g.get_equivalents(200) == [100]

    def test_unrelated_returns_empty(self):
        assert self.g.get_equivalents(999) == []


class TestRelationshipGraphComplement:
    def setup_method(self):
        self.g = RelationshipGraph([make_rel(1, 100, 200, "COMPLEMENT")])

    def test_forward(self):
        assert self.g.get_complement(100) == [200]

    def test_reverse_symmetric(self):
        assert self.g.get_complement(200) == [100]


class TestRelationshipGraphImpliesDirectional:
    def setup_method(self):
        self.g = RelationshipGraph([make_rel(1, 100, 200, "IMPLIES")])

    def test_forward(self):
        assert self.g.get_related(100, "IMPLIES") == [200]

    def test_reverse_is_empty(self):
        assert self.g.get_related(200, "IMPLIES") == []


class TestRelationshipGraphConfidenceFiltering:
    def setup_method(self):
        self.g = RelationshipGraph(
            [
                make_rel(1, 100, 200, "EQUIVALENT", confidence=0.9),
                make_rel(2, 100, 300, "EQUIVALENT", confidence=0.5),
            ]
        )

    def test_no_filter(self):
        result = self.g.get_equivalents(100)
        assert set(result) == {200, 300}

    def test_high_threshold_excludes_low(self):
        result = self.g.get_equivalents(100, min_confidence=0.8)
        assert result == [200]

    def test_threshold_inclusive(self):
        result = self.g.get_equivalents(100, min_confidence=0.9)
        assert result == [200]

    def test_threshold_excludes_all(self):
        result = self.g.get_equivalents(100, min_confidence=1.0)
        assert result == []


class TestRelationshipGraphConstructorConfidence:
    def test_constructor_min_confidence_excludes(self):
        g = RelationshipGraph(
            [
                make_rel(1, 100, 200, "EQUIVALENT", confidence=0.9),
                make_rel(2, 100, 300, "EQUIVALENT", confidence=0.5),
            ],
            min_confidence=0.8,
        )
        assert g.get_equivalents(100) == [200]
        assert g.get_equivalents(300) == []


class TestRelationshipGraphEquivalentPairs:
    def test_no_duplicates(self):
        g = RelationshipGraph(
            [
                make_rel(1, 100, 200, "EQUIVALENT"),
                make_rel(2, 200, 300, "EQUIVALENT"),
            ]
        )
        pairs = g.get_equivalent_pairs()
        assert len(pairs) == 2
        assert (100, 200) in pairs
        assert (200, 300) in pairs

    def test_pair_normalized(self):
        g = RelationshipGraph([make_rel(1, 200, 100, "EQUIVALENT")])
        pairs = g.get_equivalent_pairs()
        assert pairs == [(100, 200)]

    def test_confidence_filter(self):
        g = RelationshipGraph(
            [
                make_rel(1, 100, 200, "EQUIVALENT", confidence=0.9),
                make_rel(2, 300, 400, "EQUIVALENT", confidence=0.5),
            ]
        )
        pairs = g.get_equivalent_pairs(min_confidence=0.8)
        assert pairs == [(100, 200)]


class TestRelationshipGraphMultipleTypes:
    def setup_method(self):
        self.g = RelationshipGraph(
            [
                make_rel(1, 100, 200, "EQUIVALENT"),
                make_rel(2, 100, 300, "COMPLEMENT"),
                make_rel(3, 100, 400, "IMPLIES"),
            ]
        )

    def test_equivalents_only_returns_equivalent(self):
        assert self.g.get_equivalents(100) == [200]

    def test_complement_only_returns_complement(self):
        assert self.g.get_complement(100) == [300]

    def test_implies_only_returns_implies(self):
        assert self.g.get_related(100, "IMPLIES") == [400]
