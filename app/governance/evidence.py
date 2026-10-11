"""Deterministic review of evidence-bound subagent research results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.rag.evidence import detect_fact_conflicts
from app.schemas.planning import Evidence
from app.schemas.research import Claim, EvidenceBoundCandidate, ResearchConflict, SubagentResponse


SOURCE_RANKS: dict[str, int] = {
    "official": 0,
    "weather": 1,
    "weather_mcp": 1,
    "transport": 2,
    "transport_mcp": 2,
    "hotel": 3,
    "hotel_mcp": 3,
    "local": 4,
    "local_rag": 4,
    "rag": 4,
    "search": 5,
    "search_mcp": 5,
    "web": 5,
    "fallback": 6,
}


class ReviewedResearch(BaseModel):
    """Governed research payload safe for downstream planning use."""

    claims: list[Claim] = Field(default_factory=list)
    candidates: list[EvidenceBoundCandidate] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    conflicts: list[ResearchConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    responses: list[SubagentResponse] = Field(default_factory=list)


class EvidenceGovernanceService:
    """Validate, deduplicate, and filter evidence-bound subagent output."""

    def __init__(self, *, now: datetime | None = None):
        self._now = now

    def review(self, results: Iterable[SubagentResponse | Mapping[str, Any]]) -> ReviewedResearch:
        warnings: list[str] = []
        responses = [
            result if isinstance(result, SubagentResponse) else SubagentResponse.model_validate(result)
            for result in results
        ]
        usable_evidence, id_remap = self._review_evidence(responses, warnings)
        usable_ids = {item.id for item in usable_evidence if item.id}
        evidence_by_id = {item.id: item for item in usable_evidence if item.id}

        claims: list[Claim] = []
        seen_claims: set[tuple[str, tuple[str, ...]]] = set()
        candidates: list[EvidenceBoundCandidate] = []
        seen_candidates: set[tuple[str, str, tuple[str, ...]]] = set()
        conflicts: list[ResearchConflict] = []
        reviewed_responses: list[SubagentResponse] = []

        for response in responses:
            warnings.extend(response.warnings)
            claim_sources = list(response.claims)
            candidate_sources = list(response.candidates)
            report = response.research_report
            if response.research_report is not None:
                warnings.extend(response.research_report.warnings)
                claim_sources.extend(response.research_report.claims)
                conflicts.extend(response.research_report.conflicts)

            response_claims: list[Claim] = []
            response_candidates: list[EvidenceBoundCandidate] = []
            response_candidate_keys: set[tuple[str, str, tuple[str, ...]]] = set()

            for claim in claim_sources:
                remapped_ids = self._remap_ids(claim.evidence_ids, id_remap, usable_ids)
                if not remapped_ids:
                    warnings.append(
                        f"Dropped unbound claim for task {response.task_id}: {claim.text[:80]}"
                    )
                    continue
                key = (claim.text, tuple(remapped_ids))
                if key in seen_claims:
                    continue
                seen_claims.add(key)
                governed_claim = claim.model_copy(update={"evidence_ids": remapped_ids})
                claims.append(governed_claim)
                response_claims.append(governed_claim)

            for candidate in candidate_sources:
                remapped_ids = self._remap_ids(candidate.evidence_ids, id_remap, usable_ids)
                if not remapped_ids:
                    warnings.append(
                        f"Dropped unbound candidate for task {response.task_id}: {candidate.name[:80]}"
                    )
                    continue
                key = (candidate.name, candidate.category, tuple(remapped_ids))
                governed_candidate = candidate.model_copy(update={"evidence_ids": remapped_ids})
                if key not in seen_candidates:
                    seen_candidates.add(key)
                    candidates.append(governed_candidate)
                if key not in response_candidate_keys:
                    response_candidate_keys.add(key)
                    response_candidates.append(governed_candidate)

            response_evidence: list[Evidence] = []
            seen_response_evidence: set[str] = set()
            response_evidence_sources = list(response.evidence)
            if report is not None:
                response_evidence_sources.extend(report.evidence)
            for evidence in response_evidence_sources:
                mapped_id = id_remap.get(evidence.id)
                if mapped_id in usable_ids and mapped_id not in seen_response_evidence:
                    response_evidence.append(evidence_by_id[mapped_id])
                    seen_response_evidence.add(mapped_id)

            governed_report = None
            if report is not None:
                report_claims = [claim for claim in response_claims if claim in report.claims]
                governed_report = report.model_copy(
                    update={
                        "claims": report_claims,
                        "evidence": [item for item in response_evidence if item in report.evidence],
                    }
                )
            reviewed_responses.append(
                response.model_copy(
                    update={
                        "claims": [claim for claim in response_claims if claim in response.claims],
                        "candidates": response_candidates,
                        "evidence": [item for item in response_evidence if item in response.evidence],
                        "research_report": governed_report,
                    }
                )
            )

        metadata_conflicts = self._metadata_conflicts(usable_evidence)
        existing_conflict_keys = {conflict.fact_key for conflict in conflicts}
        for conflict in metadata_conflicts:
            if conflict.fact_key not in existing_conflict_keys:
                conflicts.append(conflict)
        if conflicts:
            warnings.append("evidence_conflict:unresolved")

        return ReviewedResearch(
            claims=claims,
            candidates=candidates,
            evidence=usable_evidence,
            conflicts=conflicts,
            warnings=list(dict.fromkeys(warnings)),
            responses=reviewed_responses,
        )

    def _review_evidence(
        self,
        responses: list[SubagentResponse],
        warnings: list[str],
    ) -> tuple[list[Evidence], dict[str, str | None]]:
        id_remap: dict[str, str | None] = {}
        evidence_by_key: dict[str, Evidence] = {}
        order_by_key: dict[str, int] = {}
        key_by_evidence_id: dict[str, str] = {}
        evidence_ids_by_key: dict[str, set[str]] = {}
        now = self._as_aware_utc(self._now or datetime.now(timezone.utc))

        for response in responses:
            evidence_items = list(response.evidence)
            if response.research_report is not None:
                evidence_items.extend(response.research_report.evidence)

            for evidence in evidence_items:
                if not evidence.id:
                    warnings.append(f"Dropped evidence without id for task {response.task_id}.")
                    continue
                key = self._dedupe_key(evidence)
                valid_from = (
                    self._as_aware_utc(evidence.valid_from)
                    if evidence.valid_from is not None
                    else None
                )
                valid_until = (
                    self._as_aware_utc(evidence.valid_until)
                    if evidence.valid_until is not None
                    else None
                )
                if valid_from is not None or valid_until is not None:
                    evidence = evidence.model_copy(
                        update={
                            "valid_from": valid_from,
                            "valid_until": valid_until,
                        }
                    )
                if valid_from is not None and valid_from > now:
                    id_remap[evidence.id] = None
                    warnings.append(f"Dropped future evidence {evidence.id} for task {response.task_id}.")
                    continue
                if valid_until is not None and valid_until < now:
                    id_remap[evidence.id] = None
                    warnings.append(f"Dropped expired evidence {evidence.id} for task {response.task_id}.")
                    continue
                if self._is_external(evidence) and not evidence.source_url:
                    id_remap[evidence.id] = None
                    warnings.append(
                        f"Dropped external evidence without source URL {evidence.id} for task {response.task_id}."
                    )
                    continue
                if evidence.id in key_by_evidence_id and key_by_evidence_id[evidence.id] != key:
                    id_remap[evidence.id] = None
                    warnings.append(f"Dropped conflicting duplicate evidence id {evidence.id}.")
                    continue

                key_by_evidence_id[evidence.id] = key
                existing = evidence_by_key.get(key)
                if existing is None:
                    evidence_by_key[key] = evidence
                    order_by_key[key] = len(order_by_key)
                    evidence_ids_by_key[key] = {evidence.id}
                    id_remap[evidence.id] = evidence.id
                    continue

                evidence_ids_by_key.setdefault(key, set()).add(evidence.id)
                preferred = self._preferred_evidence(existing, evidence)
                evidence_by_key[key] = preferred
                kept_id = preferred.id
                for duplicate_id in evidence_ids_by_key[key]:
                    id_remap[duplicate_id] = kept_id
                warnings.append(
                    f"Deduplicated evidence {evidence.id} into {kept_id} for task {response.task_id}."
                )

        ordered_keys = sorted(order_by_key, key=order_by_key.__getitem__)
        return [evidence_by_key[key] for key in ordered_keys], id_remap

    @staticmethod
    def _dedupe_key(evidence: Evidence) -> str:
        normalized_content = " ".join(evidence.content.split()).lower()
        if evidence.source_url:
            url = evidence.source_url.strip().lower()
            metadata = evidence.metadata
            # 接口类来源（高德天气、路线）每条证据都指向同一个文档地址，内容却各不相同；
            # 只按地址去重会把 4 天的天气合并成 1 天。
            if metadata.get("source_type") == "api" or metadata.get("provider") == "amap":
                return f"url:{url}|content:{normalized_content}"
            return f"url:{url}"
        return f"content:{normalized_content}"

    @staticmethod
    def _is_external(evidence: Evidence) -> bool:
        metadata = evidence.metadata
        source_type = str(metadata.get("source_type", "")).strip().lower()
        if source_type in {"mock_markdown", "local", "local_rag", "rag", "synthetic", "dataset"}:
            return False
        if metadata.get("is_external") is True:
            return True
        if source_type in {"external", "web", "api", "mcp"}:
            return True
        if str(metadata.get("provider", "")).strip().lower() in {
            "amap",
            "tavily",
            "web",
            "mcp",
        }:
            return True
        # A URL is the strongest source-level signal that the item came from outside local RAG.
        return bool(evidence.source_url)

    @staticmethod
    def _metadata_conflicts(items: list[Evidence]) -> list[ResearchConflict]:
        """委托给 app.rag.evidence 的唯一实现，避免判定口径三处各说一套。"""
        return detect_fact_conflicts(items)

    @staticmethod
    def _source_rank(evidence: Evidence) -> tuple[int, float]:
        metadata = evidence.metadata
        candidates = [
            str(metadata.get("provider", "")).strip().lower(),
            str(metadata.get("source_type", "")).strip().lower(),
            evidence.source.strip().lower(),
        ]
        for candidate in candidates:
            if candidate in SOURCE_RANKS:
                return (SOURCE_RANKS[candidate], -evidence.confidence)
            for source_name, rank in SOURCE_RANKS.items():
                if source_name in candidate:
                    return (rank, -evidence.confidence)
        return (99, -evidence.confidence)

    @staticmethod
    def _as_aware_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _preferred_evidence(self, current: Evidence, candidate: Evidence) -> Evidence:
        if self._source_rank(candidate) < self._source_rank(current):
            return candidate
        return current

    @staticmethod
    def _remap_ids(
        evidence_ids: list[str],
        id_remap: dict[str, str | None],
        usable_ids: set[str | None],
    ) -> list[str]:
        remapped: list[str] = []
        for evidence_id in evidence_ids:
            mapped = id_remap.get(evidence_id)
            if mapped is None or mapped not in usable_ids:
                continue
            remapped.append(mapped)
        return list(dict.fromkeys(remapped))


__all__ = ["EvidenceGovernanceService", "ReviewedResearch"]
