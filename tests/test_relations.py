import random

import pytest

from gnomepy.registry.relations import EQUIVALENT, IMPLIES, Edge, close, reduce, relationship_type

DIFFERS = "RESOLUTION_SOURCE_DIFFERS"
ZONE = "DEADLINE_TIMEZONE_DIFFERS"


def implies(a, b, *flags):
    return Edge(a, b, IMPLIES, tuple(flags))


def equivalent(a, b, *flags):
    return Edge(a, b, EQUIVALENT, tuple(flags))


def ladder(first: int, strikes: int) -> tuple[list[Edge], dict[int, int]]:
    """Every pair of a Yes/No strike ladder, as derivation lists it: higher Yes implies lower Yes, lower No implies
    higher No. Yes ids first..first+strikes-1 (rising strikes), their No sides after."""
    yes = list(range(first, first + strikes))
    no = [s + strikes for s in yes]
    edges = [implies(yes[j], yes[i]) for i in range(strikes) for j in range(i + 1, strikes)]
    edges += [implies(no[i], no[j]) for i in range(strikes) for j in range(i + 1, strikes)]
    return edges, {**dict(zip(yes, no)), **dict(zip(no, yes))}


class TestClose:
    def test_implication_chains(self):
        rel = close([implies(1, 2), implies(2, 3)], {})
        assert rel[(1, 3)] == ()
        assert (3, 1) not in rel

    def test_equivalence_is_both_ways_and_chains(self):
        rel = close([equivalent(1, 2), implies(2, 3)], {})
        assert rel[(2, 1)] == () and rel[(1, 3)] == ()

    def test_contraposition_through_complements(self):
        rel = close([implies(1, 2)], {1: 11, 11: 1, 2: 12, 12: 2})
        assert rel[(12, 11)] == ()

    def test_a_flagged_row_takes_unflagged_paths_on_both_sides(self):
        rel = close([implies(1, 2), implies(2, 3, DIFFERS), implies(3, 4)], {})
        assert rel[(1, 4)] == (DIFFERS,)

    def test_flagged_rows_never_chain(self):
        rel = close([implies(1, 2, ZONE), implies(2, 3, ZONE)], {})
        assert (1, 3) not in rel

    def test_the_route_with_fewest_flags_wins(self):
        rel = close([implies(1, 3, DIFFERS), implies(1, 2), implies(2, 3)], {})
        assert rel[(1, 3)] == ()

    def test_relationship_type_reads_a_pair(self):
        rel = close([equivalent(1, 2, DIFFERS), implies(2, 3)], {})
        assert relationship_type(rel, 2, 1) == (EQUIVALENT, 1, 2, (DIFFERS,))
        assert relationship_type(rel, 3, 2) == (IMPLIES, 2, 3, ())
        assert relationship_type(rel, 1, 9) is None


class TestReduce:
    def test_a_yes_no_ladder_keeps_one_row_per_step(self):
        edges, complements = ladder(1, 40)
        rel = close(edges, complements)
        rows = reduce(rel, complements)
        assert len(edges) == 1560
        assert len(rows) == 39
        assert close(rows, complements) == rel

    def test_the_same_ladder_on_two_venues_adds_one_row_per_strike(self):
        k_edges, k_comp = ladder(1, 10)
        p_edges, p_comp = ladder(101, 10)
        same = [equivalent(i, 100 + i) for i in range(1, 21)]
        complements = {**k_comp, **p_comp}
        rel = close(k_edges + p_edges + same, complements)
        rows = reduce(rel, complements)
        assert close(rows, complements) == rel
        assert len(rows) == 9 + 10

    def test_a_flagged_cross_venue_ladder_keeps_one_flagged_row_per_strike(self):
        k_edges, k_comp = ladder(1, 10)
        p_edges, p_comp = ladder(101, 10)
        same = [equivalent(i, 100 + i, DIFFERS) for i in range(1, 21)]
        complements = {**k_comp, **p_comp}
        rel = close(k_edges + p_edges + same, complements)
        rows = reduce(rel, complements)
        assert close(rows, complements) == rel
        assert sum(1 for r in rows if r.flags) == 10
        assert len(rows) == 9 + 9 + 10

    def test_rows_are_minimal(self):
        edges, complements = ladder(1, 6)
        edges += [equivalent(1, 50), implies(50, 60, ZONE), implies(60, 61)]
        rel = close(edges, complements)
        rows = reduce(rel, complements)
        assert close(rows, complements) == rel
        for i in range(len(rows)):
            assert close(rows[:i] + rows[i + 1:], complements) != rel

    @pytest.mark.parametrize("seed", range(1000))
    def test_random_relations_round_trip_with_no_spare_row(self, seed):
        rng = random.Random(seed)
        n = rng.randint(2, 14)
        ids = list(range(1, n + 1))
        complements: dict[int, int] = {}
        pool = ids[:]
        rng.shuffle(pool)
        while len(pool) >= 2 and rng.random() < 0.7:
            a, b = pool.pop(), pool.pop()
            complements[a], complements[b] = b, a
        edges = []
        for _ in range(rng.randint(1, 3 * n)):
            a, b = rng.sample(ids, 2)
            flags = rng.choice([(), (), (), (DIFFERS,), (ZONE,), (DIFFERS, ZONE)])
            edges.append(Edge(a, b, rng.choice([IMPLIES, IMPLIES, EQUIVALENT]), flags))
        rel = close(edges, complements)
        rows = reduce(rel, complements)
        assert close(rows, complements) == rel
        if any(pair in rel for pair in complements.items()):
            # A security equivalent to or implying its own complement is a contradiction derivation never produces;
            # its rows still round-trip but needn't be minimal.
            return
        for i in range(len(rows)):
            assert close(rows[:i] + rows[i + 1:], complements) != rel, rows[i]
