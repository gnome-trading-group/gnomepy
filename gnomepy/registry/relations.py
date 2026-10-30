"""How stored relationship rows compose into the complete relation, and the fewest rows that rebuild a relation.

The classifier stores only the rows nothing else rebuilds; every consumer rebuilds the rest with `close`, so both
sides agree on the rules:

1. IMPLIES is transitive; EQUIVALENT is IMPLIES both ways.
2. Contraposition through complements (a BINARY event's two securities): A ⇒ B gives not-B ⇒ not-A.
3. Unflagged rows compose freely.
4. A flagged row composes only with unflagged paths on either side, never with another flagged row, and the result
   carries that row's flags. Rows flagged for a time-zone or snapshot-time tolerance are not transitive, and source
   flags stay exact because the unflagged ends share the flagged row's sources.
5. A pair reachable several ways takes the route with the fewest flags (unflagged first).
6. Manual rows compose like unflagged rows.

Equivalent securities form a class; all members of a class relate to everything the same way, so the work is done on
classes, with reachability as bitsets over them.
"""

__all__ = ["Edge", "Relation", "close", "reduce", "relationship_type"]

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

EQUIVALENT = "EQUIVALENT"
IMPLIES = "IMPLIES"


@dataclass(frozen=True)
class Edge:
    security_id_a: int
    security_id_b: int
    relationship_type: str
    flags: tuple[str, ...] = ()


# a ⇒ b, for a != b, to the flags of its best route.
Relation = dict[tuple[int, int], tuple[str, ...]]


def _order(flags: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    return len(flags), flags


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _directed(edges: Iterable[Edge], complements: Mapping[int, int]):
    unflagged: set[tuple[int, int]] = set()
    flagged: set[tuple[int, int, tuple[str, ...]]] = set()
    for e in edges:
        pairs = [(e.security_id_a, e.security_id_b)]
        if e.relationship_type == EQUIVALENT:
            pairs.append((e.security_id_b, e.security_id_a))
        flags = tuple(sorted(e.flags))
        for x, y in pairs:
            directed = [(x, y)]
            if x in complements and y in complements:
                directed.append((complements[y], complements[x]))
            for a, b in directed:
                if a == b:
                    continue
                if flags:
                    flagged.add((a, b, flags))
                else:
                    unflagged.add((a, b))
    return unflagged, flagged


class _Classes:
    """Equivalence classes of `nodes` under directed reachability, and each class's descendants (itself included)."""

    def __init__(self, nodes: Iterable[int], arcs: Iterable[tuple[int, int]]):
        self.nodes = sorted(set(nodes))
        succ: dict[int, list[int]] = {n: [] for n in self.nodes}
        for a, b in arcs:
            succ[a].append(b)
        self.of: dict[int, int] = {}
        self.members: list[list[int]] = []
        # Iterative Tarjan: classes come out sinks first, so descendants are complete when a class is closed.
        index: dict[int, int] = {}
        low: dict[int, int] = {}
        stack: list[int] = []
        on_stack: set[int] = set()
        counter = 0
        for root in self.nodes:
            if root in index:
                continue
            work = [(root, 0)]
            while work:
                node, i = work.pop()
                if i == 0:
                    index[node] = low[node] = counter
                    counter += 1
                    stack.append(node)
                    on_stack.add(node)
                recurse = False
                for j in range(i, len(succ[node])):
                    nxt = succ[node][j]
                    if nxt not in index:
                        work.append((node, j + 1))
                        work.append((nxt, 0))
                        recurse = True
                        break
                    if nxt in on_stack:
                        low[node] = min(low[node], index[nxt])
                if recurse:
                    continue
                if low[node] == index[node]:
                    members = []
                    while True:
                        m = stack.pop()
                        on_stack.discard(m)
                        self.of[m] = len(self.members)
                        members.append(m)
                        if m == node:
                            break
                    self.members.append(sorted(members))
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
        self.succ: list[int] = [0] * len(self.members)
        for a, b in arcs:
            ca, cb = self.of[a], self.of[b]
            if ca != cb:
                self.succ[ca] |= 1 << cb
        self.desc: list[int] = [0] * len(self.members)
        for c in range(len(self.members)):
            mask = 1 << c
            for d in _bits(self.succ[c]):
                mask |= self.desc[d]
            self.desc[c] = mask


def _compose(unflagged: set[tuple[int, int]], flagged: set[tuple[int, int, tuple[str, ...]]]) -> Relation:
    nodes = {n for pair in unflagged for n in pair} | {n for a, b, _ in flagged for n in (a, b)}
    classes = _Classes(nodes, unflagged)
    rel: Relation = {}
    for c, members in enumerate(classes.members):
        for d in _bits(classes.desc[c]):
            for a in members:
                for b in classes.members[d]:
                    if a != b:
                        rel[(a, b)] = ()
    by_flags: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for a, b, flags in flagged:
        by_flags.setdefault(flags, []).append((classes.of[a], classes.of[b]))
    for flags in sorted(by_flags, key=_order):
        seeded = [0] * len(classes.members)
        for ca, cb in by_flags[flags]:
            seeded[ca] |= classes.desc[cb]
        for c, members in enumerate(classes.members):
            reach = 0
            for x in _bits(classes.desc[c]):
                reach |= seeded[x]
            for d in _bits(reach):
                for a in members:
                    for b in classes.members[d]:
                        if a != b and ((a, b) not in rel or _order(flags) < _order(rel[(a, b)])):
                            rel[(a, b)] = flags
    return rel


def close(edges: Iterable[Edge], complements: Mapping[int, int]) -> Relation:
    """The complete relation the rows give under the rules above."""
    unflagged, flagged = _directed(edges, complements)
    while True:
        rel = _compose(unflagged, flagged)
        # A composed pair has a contrapositive too, even when a security on its route has no complement; adding them
        # can open new routes, so compose again until nothing new appears.
        grew = False
        for (a, b), flags in list(rel.items()):
            if a not in complements or b not in complements:
                continue
            mirror = (complements[b], complements[a])
            if mirror[0] == mirror[1] or (mirror in rel and _order(rel[mirror]) <= _order(flags)):
                continue
            grew = True
            if flags:
                flagged.add((*mirror, flags))
            else:
                unflagged.add(mirror)
        if not grew:
            return rel


def relationship_type(rel: Relation, a: int, b: int) -> tuple[str, int, int, tuple[str, ...]] | None:
    """The relationship between a and b as a row would state it: (type, antecedent or lower id, other, flags)."""
    forward, backward = rel.get((a, b)), rel.get((b, a))
    if forward is not None and backward is not None:
        lo, hi = min(a, b), max(a, b)
        return EQUIVALENT, lo, hi, tuple(sorted(set(forward) | set(backward)))
    if forward is not None:
        return IMPLIES, a, b, forward
    if backward is not None:
        return IMPLIES, b, a, backward
    return None


def _mirror(edge: Edge, complements: Mapping[int, int]) -> Edge | None:
    a, b = edge.security_id_a, edge.security_id_b
    if a not in complements or b not in complements:
        return None
    if edge.relationship_type == EQUIVALENT:
        lo, hi = sorted((complements[a], complements[b]))
        return Edge(lo, hi, EQUIVALENT, edge.flags)
    return Edge(complements[b], complements[a], IMPLIES, edge.flags)


def _key(edge: Edge) -> tuple:
    return min(edge.security_id_a, edge.security_id_b), max(edge.security_id_a, edge.security_id_b), \
        edge.relationship_type, edge.flags


def reduce(rel: Relation, complements: Mapping[int, int]) -> list[Edge]:
    """Rows that `close` rebuilds into `rel`, given the same complements, none of which the others rebuild. `rel` must
    be closed (a result of `close`); the caller checks the round trip."""
    nodes = {n for pair in rel for n in pair}
    complements = {a: b for a, b in complements.items() if a in nodes and b in nodes}
    classes = _Classes(nodes, [p for p, flags in rel.items() if not flags])
    count = len(classes.members)

    # Representatives: a class and its mirror class (the complements of its members) pick mirrored members, so their
    # rows mirror each other and one copy can be dropped.
    rep: dict[int, int] = {}
    for c in sorted(range(count), key=lambda c: classes.members[c][0]):
        if c in rep:
            continue
        with_complement = [m for m in classes.members[c] if m in complements]
        rep[c] = min(with_complement) if with_complement else classes.members[c][0]
        if rep[c] in complements:
            mirror_class = classes.of[complements[rep[c]]]
            if mirror_class not in rep and mirror_class != c:
                rep[mirror_class] = complements[rep[c]]

    rows: list[Edge] = []
    for c, members in enumerate(classes.members):
        rows.extend(Edge(*sorted((rep[c], m)), EQUIVALENT) for m in members if m != rep[c])
    strict = [classes.desc[c] & ~(1 << c) for c in range(count)]
    covers: list[int] = []
    for c in range(count):
        beyond = 0
        for d in _bits(strict[c]):
            beyond |= strict[d]
        covers.append(strict[c] & ~beyond)
        rows.extend(Edge(rep[c], rep[d], IMPLIES) for d in _bits(covers[c]))

    # Flagged pairs, between classes: a row is needed exactly where no neighbouring class (one step up from the
    # antecedent, or one step down from the consequent) relates with the same flags, since that one's row would
    # already rebuild it.
    covered_by: list[int] = [0] * count
    for c in range(count):
        for d in _bits(covers[c]):
            covered_by[d] |= 1 << c
    between: dict[tuple[int, int], tuple[str, ...]] = {}
    for (a, b), flags in rel.items():
        if flags:
            between[(classes.of[a], classes.of[b])] = flags
    needed: dict[tuple[int, int], tuple[str, ...]] = {}
    for (c, d), flags in between.items():
        if any(between.get((up, d)) == flags for up in _bits(covers[c])):
            continue
        if any(between.get((c, down)) == flags for down in _bits(covered_by[d])):
            continue
        needed[(c, d)] = flags
    for (c, d), flags in sorted(needed.items()):
        if (d, c) in needed and needed[(d, c)] == flags:
            if c < d:
                rows.append(Edge(*sorted((rep[c], rep[d])), EQUIVALENT, flags))
            continue
        rows.append(Edge(rep[c], rep[d], IMPLIES, flags))

    kept = set(rows)
    out = []
    for row in sorted(rows, key=_key):
        mirror = _mirror(row, complements)
        if mirror is not None and mirror != row and mirror in kept and _key(mirror) < _key(row):
            continue
        out.append(row)
    return _prune(out, rel, complements)


def _prune(rows: list[Edge], rel: Relation, complements: Mapping[int, int]) -> list[Edge]:
    """Drops rows the others rebuild through contrapositives of composed pairs, which the class structure above can't
    see. Only a flagged row, or one whose ends both have complements, can be rebuilt another way; the others are
    covering steps of the class order and always needed. Highest ids are tried first, so lower ids keep their rows."""
    kept = list(rows)
    for row in sorted(rows, key=_key, reverse=True):
        if not row.flags and not (row.security_id_a in complements and row.security_id_b in complements):
            continue
        rest = [r for r in kept if r != row]
        if close(rest, complements) == rel:
            kept = rest
    return kept
