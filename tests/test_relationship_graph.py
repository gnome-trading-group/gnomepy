import pytest

from gnomepy.registry.graph import RelationshipGraph
from gnomepy.registry.types import ContractRelationship, ContractType, EventContract, Security


def make_rel(rel_id, id_a, id_b, rel_type, method="derived", flags=(), complement_a=None, complement_b=None):
    return ContractRelationship(
        relationship_id=rel_id,
        security_id_a=id_a,
        security_id_b=id_b,
        relationship_type=rel_type,
        method=method,
        date_created="2025-01-01T00:00:00",
        flags=list(flags),
        complement_a=complement_a,
        complement_b=complement_b,
    )


def make_ec(event_id, security_id):
    return EventContract(
        event_contract_id=security_id,
        event_id=event_id,
        security_id=security_id,
        outcome_label=str(security_id),
        date_created="2025-01-01T00:00:00",
    )


def make_security(security_id, contract_type):
    return Security(
        security_id=security_id, symbol=str(security_id), type=0, contract_type=contract_type, asset_class=5,
        base_currency_id=None, quote_currency_id=None, settle_currency_id=None, inverse=False, is_quanto=False,
        expiry=None, strike_price=None, active=True, underlying_security_id=None, description=None,
        date_modified="2025-01-01T00:00:00", date_created="2025-01-01T00:00:00",
    )


class TestRelationshipGraphEmpty:
    def test_get_equivalents_empty(self):
        g = RelationshipGraph([])
        assert g.get_equivalents(1) == []

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


class TestRelationshipGraphImpliesDirectional:
    def setup_method(self):
        self.g = RelationshipGraph([make_rel(1, 100, 200, "IMPLIES")])

    def test_forward(self):
        assert self.g.get_related(100, "IMPLIES") == [200]

    def test_reverse_is_empty(self):
        assert self.g.get_related(200, "IMPLIES") == []



class TestRelationshipGraphEquivalentPairs:
    def test_equivalence_chains_into_every_pair_once(self):
        g = RelationshipGraph(
            [
                make_rel(1, 100, 200, "EQUIVALENT"),
                make_rel(2, 200, 300, "EQUIVALENT"),
            ]
        )
        assert g.get_equivalent_pairs() == [(100, 200), (100, 300), (200, 300)]

    def test_pair_normalized(self):
        g = RelationshipGraph([make_rel(1, 200, 100, "EQUIVALENT")])
        pairs = g.get_equivalent_pairs()
        assert pairs == [(100, 200)]


class TestRelationshipGraphMultipleTypes:
    def setup_method(self):
        self.g = RelationshipGraph(
            [
                make_rel(1, 100, 200, "EQUIVALENT"),
                make_rel(2, 100, 400, "IMPLIES"),
            ]
        )

    def test_equivalents_only_returns_equivalent(self):
        assert self.g.get_equivalents(100) == [200]

    def test_implies_only_returns_implies(self):
        assert self.g.get_related(100, "IMPLIES") == [400]


class TestRelationshipGraphImpliesMethods:
    def setup_method(self):
        self.g = RelationshipGraph(
            [
                make_rel(1, 100, 200, "IMPLIES"),
                make_rel(2, 100, 300, "IMPLIES"),
                make_rel(3, 400, 200, "IMPLIES"),
            ]
        )

    def test_get_implies_returns_consequents(self):
        assert set(self.g.get_implies(100)) == {200, 300}


    def test_get_implies_unknown_returns_empty(self):
        assert self.g.get_implies(999) == []

    def test_get_implied_by_returns_antecedents(self):
        assert set(self.g.get_implied_by(200)) == {100, 400}


    def test_get_implied_by_unknown_returns_empty(self):
        assert self.g.get_implied_by(999) == []

    def test_get_implies_pairs_all(self):
        pairs = self.g.get_implies_pairs()
        assert len(pairs) == 3
        assert (100, 200) in pairs
        assert (100, 300) in pairs
        assert (400, 200) in pairs


    def test_implies_is_directional(self):
        assert self.g.get_implies(200) == []
        assert self.g.get_implies(300) == []


class TestRelationshipGraphEventStructure:
    def setup_method(self):
        self.g = RelationshipGraph(
            [],
            event_contracts=[
                make_ec(1, 10), make_ec(1, 11),
                make_ec(2, 20), make_ec(2, 21), make_ec(2, 22),
                make_ec(3, 30), make_ec(3, 31),
                make_ec(4, 40),
            ],
            contract_types={
                10: ContractType.BINARY, 11: ContractType.BINARY,
                20: ContractType.MULTI_OUTCOME, 21: ContractType.MULTI_OUTCOME, 22: ContractType.MULTI_OUTCOME,
                30: ContractType.MULTI_OUTCOME, 31: ContractType.MULTI_OUTCOME,
                40: ContractType.BINARY,
            },
        )

    def test_binary_sides_are_complements(self):
        assert self.g.get_complement(10) == 11
        assert self.g.get_complement(11) == 10

    def test_multi_outcome_has_no_complement(self):
        # Two outcomes of an exclusive event can both resolve No, so neither is the other's complement.
        assert self.g.get_complement(30) is None
        assert self.g.get_complement(20) is None

    def test_lone_binary_side_has_no_complement(self):
        assert self.g.get_complement(40) is None

    def test_multi_outcome_siblings_exclude(self):
        assert set(self.g.get_exclusive(20)) == {21, 22}
        assert self.g.get_exclusive(30) == [31]

    def test_binary_sides_are_not_listed_as_exclusive(self):
        assert self.g.get_exclusive(10) == []

    def test_event_securities(self):
        assert self.g.get_event_securities() == {10, 11, 20, 21, 22, 30, 31, 40}

    def test_unknown_contract_types_give_no_structure(self):
        g = RelationshipGraph([], event_contracts=[make_ec(1, 10), make_ec(1, 11)])
        assert g.get_complement(10) is None
        assert g.get_exclusive(10) == []


class _StubRegistry:
    def __init__(self, event_contracts, securities, relationships):
        self._event_contracts = event_contracts
        self._securities = {s.security_id: s for s in securities}
        self._relationships = relationships

    def get_event_contracts(self, *, event_id):
        return [ec for ec in self._event_contracts if ec.event_id == event_id]

    def get_security(self, *, security_id):
        return [self._securities[security_id]] if security_id in self._securities else []

    def get_connected_relationships(self, *, security_ids):
        reached, frontier, rows = set(security_ids), list(security_ids), {}
        while frontier:
            sid = frontier.pop()
            for r in self._relationships:
                if sid in (r.security_id_a, r.security_id_b):
                    rows[r.relationship_id] = r
                    for other in (r.security_id_a, r.security_id_b):
                        if other not in reached:
                            reached.add(other)
                            frontier.append(other)
        return list(rows.values())


class TestRelationshipGraphLoad:
    def setup_method(self):
        self.client = _StubRegistry(
            event_contracts=[make_ec(1, 10), make_ec(1, 11), make_ec(9, 90)],
            securities=[make_security(10, ContractType.BINARY), make_security(11, ContractType.BINARY)],
            relationships=[
                make_rel(1, 10, 90, "EQUIVALENT"),
                make_rel(2, 11, 91, "EQUIVALENT", flags=["RESOLUTION_SOURCE_DIFFERS"]),
                make_rel(3, 10, 11, "IMPLIES"),
            ],
        )

    def test_reads_structure_and_relationships_to_other_events(self):
        g = RelationshipGraph.load(self.client, [1])
        assert g.get_complement(10) == 11
        assert g.get_equivalents(10) == [90]
        assert g.get_event_securities() == {10, 11}

    def test_relationship_touching_two_loaded_securities_is_kept_once(self):
        g = RelationshipGraph.load(self.client, [1])
        assert [r.relationship_id for r in g.get_stored()].count(3) == 1

    def test_flagged_relationships_left_out_by_default(self):
        assert RelationshipGraph.load(self.client, [1]).get_equivalents(11) == []

    def test_flagged_relationships_kept_when_asked(self):
        assert RelationshipGraph.load(self.client, [1], allow_flagged=True).get_equivalents(11) == [91]


class TestRelationshipGraphCompleteRelation:
    def test_a_ladder_stored_as_steps_answers_every_pair(self):
        g = RelationshipGraph([make_rel(1, 103, 102, "IMPLIES"), make_rel(2, 102, 101, "IMPLIES")])
        assert g.get_implies(103) == [101, 102]
        assert g.get_implied_by(101) == [102, 103]
        assert (103, 101) in g.get_implies_pairs()

    def test_no_sides_follow_from_complements_on_the_rows(self):
        g = RelationshipGraph([make_rel(1, 103, 102, "IMPLIES", complement_a=203, complement_b=202)])
        assert g.get_complement(203) == 103
        assert g.get_implies(202) == [203]

    def test_flags_apply_to_the_composed_route(self):
        rows = [make_rel(1, 1, 2, "IMPLIES"), make_rel(2, 2, 3, "EQUIVALENT", flags=["RESOLUTION_SOURCE_DIFFERS"])]
        assert RelationshipGraph(rows).get_flags(1, 3) == ("RESOLUTION_SOURCE_DIFFERS",)
        assert RelationshipGraph(rows, allow_flagged=False).get_implies(1) == [2]

    def test_explain_names_the_rows_behind_an_implied_pair(self):
        g = RelationshipGraph([make_rel(7, 3, 2, "IMPLIES"), make_rel(8, 2, 1, "IMPLIES")])
        assert g.explain(3, 1) == [7, 8]
        assert g.explain(1, 3) is None
