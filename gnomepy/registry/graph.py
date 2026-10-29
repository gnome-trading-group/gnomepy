__all__ = ["RelationshipGraph"]

from collections import defaultdict

from gnomepy.registry.types import ContractRelationship

_DIRECTIONAL_TYPES = frozenset({"IMPLIES"})


class RelationshipGraph:
    def __init__(self, relationships: list[ContractRelationship]) -> None:
        # adj[relationship_type][security_id] -> other security ids
        self._adj: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

        for rel in relationships:
            self._adj[rel.relationship_type][rel.security_id_a].append(rel.security_id_b)
            if rel.relationship_type not in _DIRECTIONAL_TYPES:
                self._adj[rel.relationship_type][rel.security_id_b].append(rel.security_id_a)

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
