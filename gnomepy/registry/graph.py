__all__ = ["RelationshipGraph"]

from collections import defaultdict
from collections.abc import Iterable, Mapping

from gnomepy.registry.api import RegistryClient
from gnomepy.registry.relations import EQUIVALENT, Edge, close
from gnomepy.registry.types import ContractRelationship, ContractType, EventContract


class RelationshipGraph:
    """The complete relation between securities, rebuilt from the registry's stored rows (only the fewest that
    rebuild it are stored; `relations` has the composition rules), plus the structure inside each event that is never
    stored as rows: a BINARY event's two securities are complements, and a MULTI_OUTCOME event's securities exclude
    each other.

    Every query answers from the complete relation, so a pair implied through other rows (a higher strike implying a
    strike three steps down) is returned like a stored one. With `allow_flagged` off, pairs whose best route carries a
    flag (equivalent questions settled by different sources) are left out: they can resolve differently, so they are
    not safe to trade as the same contract."""

    def __init__(
        self,
        relationships: list[ContractRelationship],
        event_contracts: Iterable[EventContract] = (),
        contract_types: Mapping[int, int] | None = None,
        allow_flagged: bool = True,
    ) -> None:
        self._stored = list(relationships)
        contract_types = contract_types or {}
        by_event: dict[int, list[int]] = defaultdict(list)
        for ec in event_contracts:
            by_event[ec.event_id].append(ec.security_id)

        self._event_securities: set[int] = set()
        self._complement: dict[int, int] = {}
        self._exclusive: dict[int, list[int]] = {}
        for sids in by_event.values():
            self._event_securities.update(sids)
            types = {contract_types.get(sid) for sid in sids}
            if len(types) != 1:
                continue
            (contract_type,) = types
            if contract_type == ContractType.BINARY and len(sids) == 2:
                self._complement[sids[0]], self._complement[sids[1]] = sids[1], sids[0]
            elif contract_type == ContractType.MULTI_OUTCOME:
                for sid in sids:
                    self._exclusive[sid] = [other for other in sids if other != sid]
        for rel in self._stored:
            for sid, other in ((rel.security_id_a, rel.complement_a), (rel.security_id_b, rel.complement_b)):
                if other is not None:
                    self._complement[sid], self._complement[other] = other, sid

        relation = close(
            [Edge(r.security_id_a, r.security_id_b, r.relationship_type, tuple(r.flags)) for r in self._stored],
            self._complement,
        )
        self._relation = {pair: flags for pair, flags in relation.items() if allow_flagged or not flags}
        self._implies: dict[int, list[int]] = defaultdict(list)
        for a, b in sorted(self._relation):
            self._implies[a].append(b)

    @classmethod
    def load(cls, client: RegistryClient, event_ids: Iterable[int], allow_flagged: bool = False) -> "RelationshipGraph":
        """The events' contracts and the complete relation around them: every stored row their securities connect to,
        however far, since an implied pair can run through securities of other events."""
        event_contracts = [ec for event_id in event_ids for ec in client.get_event_contracts(event_id=event_id)]
        security_ids = sorted({ec.security_id for ec in event_contracts})
        contract_types = {
            security.security_id: security.contract_type
            for sid in security_ids
            for security in client.get_security(security_id=sid)
        }
        relationships = client.get_connected_relationships(security_ids=security_ids) if security_ids else []
        return cls(relationships, event_contracts, contract_types, allow_flagged=allow_flagged)

    def get_event_securities(self) -> set[int]:
        return set(self._event_securities)

    def get_complement(self, security_id: int) -> int | None:
        """The other side of a BINARY event; None for anything else, since a lone or multi-outcome security has no
        tradeable complement."""
        return self._complement.get(security_id)

    def get_exclusive(self, security_id: int) -> list[int]:
        """The other securities of its MULTI_OUTCOME event. At most one pays, but all of them may not (an event can
        resolve with no listed outcome)."""
        return list(self._exclusive.get(security_id, []))

    def get_flags(self, security_id_a: int, security_id_b: int) -> tuple[str, ...] | None:
        """The flags of a ⇒ b's best route, or None when a doesn't imply b."""
        return self._relation.get((security_id_a, security_id_b))

    def get_equivalents(self, security_id: int) -> list[int]:
        return self.get_related(security_id, EQUIVALENT)

    def get_equivalent_pairs(self) -> list[tuple[int, int]]:
        return sorted((a, b) for a, b in self._relation if a < b and (b, a) in self._relation)

    def get_implies(self, security_id: int) -> list[int]:
        """Securities this one implies but isn't equivalent to."""
        return self.get_related(security_id, "IMPLIES")

    def get_implied_by(self, security_id: int) -> list[int]:
        return sorted(a for a, b in self._relation if b == security_id and (b, a) not in self._relation)

    def get_implies_pairs(self) -> list[tuple[int, int]]:
        return sorted((a, b) for a, b in self._relation if (b, a) not in self._relation)

    def get_related(self, security_id: int, relationship_type: str) -> list[int]:
        both_ways = [b for b in self._implies.get(security_id, []) if (b, security_id) in self._relation]
        if relationship_type == EQUIVALENT:
            return both_ways
        return [b for b in self._implies.get(security_id, []) if (b, security_id) not in self._relation]

    def get_stored(self) -> list[ContractRelationship]:
        """The registry's rows as loaded: the fewest that rebuild the relation."""
        return list(self._stored)

    def explain(self, security_id_a: int, security_id_b: int) -> list[int] | None:
        """The stored relationship ids along a route from a to b (a route may also step through complements, which
        reverse a row). None when a doesn't imply b, or, rarely, when it follows only from the contrapositive of a
        longer route (one through a security without a complement), which no single route over rows shows."""
        if (security_id_a, security_id_b) not in self._relation:
            return None
        steps: dict[int, list[tuple[int, int, bool]]] = defaultdict(list)
        for r in self._stored:
            arcs = [(r.security_id_a, r.security_id_b)]
            if r.relationship_type == EQUIVALENT:
                arcs.append((r.security_id_b, r.security_id_a))
            for x, y in arcs:
                steps[x].append((y, r.relationship_id, bool(r.flags)))
                if x in self._complement and y in self._complement:
                    steps[self._complement[y]].append((self._complement[x], r.relationship_id, bool(r.flags)))
        # Breadth-first over (security, whether a flagged row was used), since a route may hold one flagged row.
        start = (security_id_a, False)
        came: dict[tuple[int, bool], tuple[tuple[int, bool], int] | None] = {start: None}
        queue = [start]
        while queue:
            node = queue.pop(0)
            sid, used_flag = node
            if sid == security_id_b and node != start:
                route = []
                while came[node] is not None:
                    node, rid = came[node]
                    route.append(rid)
                return route[::-1]
            for nxt, rid, flagged in steps.get(sid, []):
                if flagged and used_flag:
                    continue
                state = (nxt, used_flag or flagged)
                if state not in came:
                    came[state] = (node, rid)
                    queue.append(state)
        return None
