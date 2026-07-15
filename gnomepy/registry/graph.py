__all__ = ["RelationshipGraph"]

from collections import defaultdict

from gnomepy.registry.types import ContractRelationship

_DIRECTIONAL_TYPES = frozenset({"IMPLIES"})


class RelationshipGraph:
    def __init__(
        self,
        relationships: list[ContractRelationship],
        min_confidence: float = 0.0,
    ) -> None:
        # adj[relationship_type][security_id] -> list of (other_security_id, confidence)
        self._adj: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for rel in relationships:
            if rel.confidence < min_confidence:
                continue
            self._adj[rel.relationship_type][rel.security_id_a].append(
                (rel.security_id_b, rel.confidence)
            )
            if rel.relationship_type not in _DIRECTIONAL_TYPES:
                self._adj[rel.relationship_type][rel.security_id_b].append(
                    (rel.security_id_a, rel.confidence)
                )

    def get_equivalents(self, security_id: int, min_confidence: float = 0.0) -> list[int]:
        return self._get_related(security_id, "EQUIVALENT", min_confidence)

    def get_complement(self, security_id: int, min_confidence: float = 0.0) -> list[int]:
        return self._get_related(security_id, "COMPLEMENT", min_confidence)

    def get_equivalent_pairs(self, min_confidence: float = 0.0) -> list[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        pairs: list[tuple[int, int]] = []
        for sid, neighbors in self._adj.get("EQUIVALENT", {}).items():
            for other_sid, confidence in neighbors:
                if confidence < min_confidence:
                    continue
                pair = (min(sid, other_sid), max(sid, other_sid))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        return pairs

    def get_related(
        self,
        security_id: int,
        relationship_type: str,
        min_confidence: float = 0.0,
    ) -> list[int]:
        return self._get_related(security_id, relationship_type, min_confidence)

    def _get_related(
        self, security_id: int, relationship_type: str, min_confidence: float
    ) -> list[int]:
        neighbors = self._adj.get(relationship_type, {}).get(security_id, [])
        return [other_id for other_id, conf in neighbors if conf >= min_confidence]
