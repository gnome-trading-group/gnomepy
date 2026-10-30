__all__ = ["RelationshipGraph"]

from collections import defaultdict
from collections.abc import Iterable, Mapping

from gnomepy.registry.api import RegistryClient
from gnomepy.registry.types import ContractRelationship, ContractType, EventContract

_DIRECTIONAL_TYPES = frozenset({"IMPLIES"})


class RelationshipGraph:
    """Stored relationships between securities, plus the structure inside each event that is never stored as rows: a
    BINARY event's two securities are complements, and a MULTI_OUTCOME event's securities exclude each other."""

    def __init__(
        self,
        relationships: list[ContractRelationship],
        event_contracts: Iterable[EventContract] = (),
        contract_types: Mapping[int, int] | None = None,
    ) -> None:
        # adj[relationship_type][security_id] -> other security ids
        self._adj: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

        for rel in relationships:
            self._adj[rel.relationship_type][rel.security_id_a].append(rel.security_id_b)
            if rel.relationship_type not in _DIRECTIONAL_TYPES:
                self._adj[rel.relationship_type][rel.security_id_b].append(rel.security_id_a)

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

    @classmethod
    def load(cls, client: RegistryClient, event_ids: Iterable[int], allow_flagged: bool = False) -> "RelationshipGraph":
        """The events' contracts and every relationship touching them, wherever the other side is. Flagged
        relationships (equivalent questions settled by different sources) are left out unless asked for: they can
        resolve differently, so they are not safe to trade as the same contract."""
        event_contracts = [ec for event_id in event_ids for ec in client.get_event_contracts(event_id=event_id)]
        security_ids = sorted({ec.security_id for ec in event_contracts})
        contract_types = {
            security.security_id: security.contract_type
            for sid in security_ids
            for security in client.get_security(security_id=sid)
        }
        relationships: dict[int, ContractRelationship] = {}
        for sid in security_ids:
            for rel in client.get_contract_relationships(security_id=sid):
                if allow_flagged or not rel.flags:
                    relationships[rel.relationship_id] = rel
        return cls(list(relationships.values()), event_contracts, contract_types)

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

    def get_equivalents(self, security_id: int) -> list[int]:
        return self.get_related(security_id, "EQUIVALENT")

    def get_equivalent_pairs(self) -> list[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        pairs: list[tuple[int, int]] = []
        for sid, neighbors in self._adj.get("EQUIVALENT", {}).items():
            for other_sid in neighbors:
                pair = (min(sid, other_sid), max(sid, other_sid))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        return pairs

    def get_implies(self, security_id: int) -> list[int]:
        return self.get_related(security_id, "IMPLIES")

    def get_implied_by(self, security_id: int) -> list[int]:
        return [sid for sid, neighbors in self._adj.get("IMPLIES", {}).items() if security_id in neighbors]

    def get_implies_pairs(self) -> list[tuple[int, int]]:
        return [(sid, other_sid) for sid, neighbors in self._adj.get("IMPLIES", {}).items() for other_sid in neighbors]

    def get_related(self, security_id: int, relationship_type: str) -> list[int]:
        return list(self._adj.get(relationship_type, {}).get(security_id, []))
