"""Tests for softrag.eval.

The metric tests check against values worked out by hand and written into the
comments, not against the implementation's own output -- a self-consistent
metric can still be the wrong metric.
"""

from __future__ import annotations

import math

import pytest

from softrag.errors import ConfigurationError
from softrag.eval import (
    EvalResult,
    average_precision,
    compare,
    comparison_table,
    evaluate,
    evaluate_engine,
    evaluate_run,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from softrag.types import Hit

# --------------------------------------------------------------------------- #
# recall / precision / hit rate
# --------------------------------------------------------------------------- #


def test_recall_at_k() -> None:
    ranked = ["a", "x", "b", "y"]
    relevant = ["a", "b", "c"]
    # 1 of 3 relevant docs in the top 2 -> 1/3.
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(1 / 3)
    # 2 of 3 in the top 4 -> 2/3.
    assert recall_at_k(ranked, relevant, 4) == pytest.approx(2 / 3)
    # k beyond the result list cannot find what was never returned.
    assert recall_at_k(ranked, relevant, 50) == pytest.approx(2 / 3)


def test_recall_is_one_when_everything_relevant_is_retrieved() -> None:
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_recall_edge_cases() -> None:
    assert recall_at_k([], ["a"], 5) == 0.0
    assert recall_at_k(["a"], [], 5) == 0.0
    assert recall_at_k(["a"], ["a"], 0) == 0.0
    # Grade 0 means not relevant, so there is nothing to recall.
    assert recall_at_k(["a"], {"a": 0}, 5) == 0.0


def test_precision_at_k_divides_by_k_not_by_the_list_length() -> None:
    # 1 relevant doc among a 2-document result list, evaluated at k=5 -> 1/5,
    # exactly as trec_eval scores an under-length run.
    assert precision_at_k(["a", "x"], ["a", "c"], 5) == pytest.approx(0.2)
    # 2 relevant in the top 4 -> 2/4.
    assert precision_at_k(["a", "x", "b", "y"], ["a", "b"], 4) == pytest.approx(0.5)


def test_precision_edge_cases() -> None:
    assert precision_at_k([], ["a"], 5) == 0.0
    assert precision_at_k(["x", "y"], ["a"], 5) == 0.0
    assert precision_at_k(["a"], ["a"], 0) == 0.0


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k(["x", "a"], ["a"], 2) == 1.0
    # The relevant document is at rank 2, outside the cutoff.
    assert hit_rate_at_k(["x", "a"], ["a"], 1) == 0.0
    assert hit_rate_at_k(["x", "y"], ["a"], 10) == 0.0
    assert hit_rate_at_k([], ["a"], 10) == 0.0


# --------------------------------------------------------------------------- #
# MRR
# --------------------------------------------------------------------------- #


def test_mrr_is_the_reciprocal_of_the_first_relevant_rank() -> None:
    assert mrr(["a", "b"], ["a"]) == 1.0
    assert mrr(["b", "a"], ["a"]) == pytest.approx(0.5)
    assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
    # Only the first relevant document counts.
    assert mrr(["x", "a", "b"], ["a", "b"]) == pytest.approx(0.5)


def test_mrr_edge_cases() -> None:
    assert mrr(["x"], ["a"]) == 0.0
    assert mrr([], ["a"]) == 0.0
    assert mrr(["a"], []) == 0.0
    # Cutoff pushes the relevant document out of scope.
    assert mrr(["x", "a"], ["a"], 1) == 0.0
    assert mrr(["x", "a"], ["a"], 2) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# nDCG
# --------------------------------------------------------------------------- #


def test_ndcg_graded_hand_computed() -> None:
    """Grades [3, 2, 3, 0, 1, 2] in rank order.

    DCG  = 3/log2(2) + 2/log2(3) + 3/log2(4) + 0/log2(5)
         + 1/log2(6) + 2/log2(7)
         = 3 + 1.2618595 + 1.5 + 0 + 0.3868528 + 0.7124144 = 6.8611267
    Ideal order is [3, 3, 2, 2, 1]:
    IDCG = 3 + 3/log2(3) + 2/log2(4) + 2/log2(5) + 1/log2(6)
         = 3 + 1.8927893 + 1 + 0.8613531 + 0.3868528 = 7.1409952
    nDCG = 6.8611267 / 7.1409952 = 0.9608077
    """
    ranked = ["d1", "d2", "d3", "d4", "d5", "d6"]
    grades = {"d1": 3, "d2": 2, "d3": 3, "d4": 0, "d5": 1, "d6": 2}
    assert ndcg_at_k(ranked, grades, 6) == pytest.approx(0.9608077, abs=1e-6)


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    grades = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)


def test_ndcg_penalises_a_reversed_ranking() -> None:
    """Grades [1, 2, 3] against the ideal [3, 2, 1].

    DCG  = 1 + 2/log2(3) + 3/log2(4) = 1 + 1.2618595 + 1.5 = 3.7618595
    IDCG = 3 + 2/log2(3) + 1/log2(4) = 3 + 1.2618595 + 0.5 = 4.7618595
    nDCG = 3.7618595 / 4.7618595 = 0.7899980
    """
    grades = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["c", "b", "a"], grades, 3) == pytest.approx(0.7899980, abs=1e-6)


def test_ndcg_binary_case() -> None:
    # One relevant document at rank 2: DCG = 1/log2(3), IDCG = 1.
    assert ndcg_at_k(["x", "a"], ["a"], 2) == pytest.approx(1 / math.log2(3))


def test_ndcg_cutoff_truncates_both_dcg_and_idcg() -> None:
    grades = {"a": 3, "b": 2, "c": 1}
    # At k=1 only the first result counts: DCG = 2, IDCG = 3.
    assert ndcg_at_k(["b", "a", "c"], grades, 1) == pytest.approx(2 / 3)


def test_ndcg_edge_cases() -> None:
    assert ndcg_at_k([], {"a": 1}, 5) == 0.0
    assert ndcg_at_k(["a"], {}, 5) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}, 5) == 0.0
    assert ndcg_at_k(["a"], {"a": 1}, 0) == 0.0
    assert ndcg_at_k(["x", "y"], {"a": 1}, 5) == 0.0


# --------------------------------------------------------------------------- #
# Average precision
# --------------------------------------------------------------------------- #


def test_average_precision_hand_computed() -> None:
    """Relevant {A, C, E}, retrieved [A, B, C, D, E].

    Relevant hits land at ranks 1, 3 and 5, with precisions 1/1, 2/3 and 3/5.
    AP = (1 + 0.6666667 + 0.6) / 3 = 0.7555556
    """
    ranked = ["A", "B", "C", "D", "E"]
    assert average_precision(ranked, ["A", "C", "E"]) == pytest.approx(
        0.7555556, abs=1e-6
    )


def test_average_precision_penalises_unretrieved_documents() -> None:
    # Only one of the three relevant documents was retrieved, at rank 1:
    # AP = (1/1) / 3.
    assert average_precision(["A", "B"], ["A", "C", "E"]) == pytest.approx(1 / 3)


def test_average_precision_edge_cases() -> None:
    assert average_precision(["A"], ["A"]) == 1.0
    assert average_precision(["x"], ["A"]) == 0.0
    assert average_precision([], ["A"]) == 0.0
    assert average_precision(["A"], []) == 0.0
    # A cutoff hides the later relevant documents.
    assert average_precision(["A", "B", "C"], ["A", "C"], 1) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# evaluate()
# --------------------------------------------------------------------------- #


def test_evaluate_averages_over_queries() -> None:
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {
        "q1": {"a": 9.0, "z": 1.0},  # relevant at rank 1 -> RR 1.0
        "q2": {"z": 9.0, "b": 1.0},  # relevant at rank 2 -> RR 0.5
    }
    scores = evaluate(qrels, run, metrics=["mrr", "recall@2"])
    assert scores["mrr"] == pytest.approx(0.75)
    assert scores["recall@2"] == pytest.approx(1.0)


def test_evaluate_scores_a_query_missing_from_the_run_as_zero() -> None:
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": {"a": 1.0}}
    assert evaluate(qrels, run, metrics=["mrr"])["mrr"] == pytest.approx(0.5)


def test_evaluate_skips_queries_with_no_relevant_documents() -> None:
    qrels = {"q1": {"a": 1}, "q2": {"b": 0}}
    run = {"q1": {"a": 1.0}, "q2": {"b": 1.0}}
    result = evaluate_run(qrels, run, metrics=["mrr"])
    assert result.queries == 1
    assert result.metrics["mrr"] == pytest.approx(1.0)


def test_evaluate_with_empty_inputs() -> None:
    assert evaluate({}, {}, metrics=["mrr", "ndcg@10"]) == {"mrr": 0.0, "ndcg@10": 0.0}
    assert evaluate({"q1": {"a": 1}}, {}, metrics=["recall@5"]) == {"recall@5": 0.0}
    assert evaluate({"q1": {"a": 1}}, {"q1": {}}, metrics=["recall@5"]) == {
        "recall@5": 0.0
    }


def test_evaluate_with_all_irrelevant_results() -> None:
    qrels = {"q1": {"a": 1}}
    run = {"q1": {"x": 3.0, "y": 2.0, "z": 1.0}}
    scores = evaluate(qrels, run, metrics=["recall@5", "precision@5", "mrr", "ndcg@5"])
    assert set(scores.values()) == {0.0}


def test_evaluate_uses_graded_relevance_for_ndcg() -> None:
    qrels = {"q1": {"a": 3, "b": 1}}
    good = evaluate(qrels, {"q1": {"a": 2.0, "b": 1.0}}, metrics=["ndcg@2"])
    bad = evaluate(qrels, {"q1": {"b": 2.0, "a": 1.0}}, metrics=["ndcg@2"])
    assert good["ndcg@2"] == pytest.approx(1.0)
    # DCG  = 1 + 3/log2(3) = 1 + 1.8927893 = 2.8927893
    # IDCG = 3 + 1/log2(3) = 3 + 0.6309298 = 3.6309298 -> nDCG = 0.7967076
    assert bad["ndcg@2"] == pytest.approx(0.7967076, abs=1e-6)


def test_evaluate_ranks_ties_deterministically() -> None:
    qrels = {"q1": {"b": 1}}
    run = {"q1": {"a": 1.0, "b": 1.0}}
    # Equal scores break on document id, so "a" ranks first and RR is 0.5.
    assert evaluate(qrels, run, metrics=["mrr"])["mrr"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Metric spec parsing
# --------------------------------------------------------------------------- #


def test_metric_specs_default_their_cutoff() -> None:
    scores = evaluate({"q1": {"a": 1}}, {"q1": {"a": 1.0}}, metrics=["recall", "ndcg"])
    assert set(scores) == {"recall@10", "ndcg@10"}


def test_metric_aliases_and_labels() -> None:
    qrels, run = {"q1": {"a": 1}}, {"q1": {"a": 1.0}}
    assert set(evaluate(qrels, run, metrics=["ap", "hitrate@3"])) == {"map", "hit_rate@3"}
    assert set(evaluate(qrels, run, metrics=["MRR@5"])) == {"mrr@5"}


def test_unknown_metric_lists_the_valid_ones() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        evaluate({}, {}, metrics=["bleu@4"])
    message = str(excinfo.value)
    assert "bleu@4" in message
    for name in ("recall", "precision", "hit_rate", "ndcg", "mrr", "map"):
        assert name in message


def test_invalid_cutoff_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        evaluate({}, {}, metrics=["recall@many"])
    with pytest.raises(ConfigurationError):
        evaluate({}, {}, metrics=["recall@0"])


# --------------------------------------------------------------------------- #
# EvalResult
# --------------------------------------------------------------------------- #


def test_eval_result_summary_is_an_aligned_table() -> None:
    result = EvalResult(
        metrics={"recall@5": 0.5, "mrr": 0.25},
        per_query={
            "q1": {"recall@5": 1.0, "mrr": 0.5},
            "q2": {"recall@5": 0.0, "mrr": 0.0},
        },
        name="hybrid",
    )
    text = result.summary()
    assert "hybrid" in text and "(2 queries)" in text
    assert "0.5000" in text
    body = [line for line in text.splitlines() if line.startswith(("recall", "mrr"))]
    # Every metric row is padded to the same width, so the scores line up.
    assert len({line.index("0.") for line in body}) == 1
    assert "q1" not in text

    detailed = result.summary(per_query=True)
    assert "q1" in detailed and "q2" in detailed


def test_eval_result_summary_without_metrics() -> None:
    assert "(no metrics)" in EvalResult().summary()


def test_eval_result_lookup_and_worst() -> None:
    result = EvalResult(
        metrics={"mrr": 0.5},
        per_query={"q1": {"mrr": 1.0}, "q2": {"mrr": 0.0}, "q3": {"mrr": 0.5}},
    )
    assert result["mrr"] == 0.5
    assert result.queries == 3
    assert result.worst("mrr", n=2) == [("q2", 0.0), ("q3", 0.5)]
    with pytest.raises(ConfigurationError):
        result.worst("ndcg@10")


# --------------------------------------------------------------------------- #
# evaluate_engine / compare
# --------------------------------------------------------------------------- #


class FakeRag:
    """A stand-in engine returning canned hits, keyed by (query, mode)."""

    def __init__(self, results: dict[tuple[str, str], list[Hit]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, dict]] = []

    def search(self, query: str, *, top_k: int = 10, **kwargs: object) -> list[Hit]:
        self.calls.append((query, top_k, dict(kwargs)))
        mode = str(kwargs.get("mode", "hybrid"))
        return self.results.get((query, mode), [])[:top_k]


def hit(doc_id: int, source: str) -> Hit:
    return Hit(id=doc_id, text=f"chunk {doc_id}", score=1.0, source=source)


def test_evaluate_engine_matches_on_source() -> None:
    rag = FakeRag(
        {
            ("what is the refund window?", "hybrid"): [
                hit(1, "faq.md"),
                hit(2, "policy.md"),
            ],
            ("who owns the data?", "hybrid"): [hit(3, "terms.md")],
        }
    )
    dataset = [
        {"query": "what is the refund window?", "relevant": ["policy.md"]},
        {"query": "who owns the data?", "relevant": ["terms.md"]},
    ]
    result = evaluate_engine(rag, dataset, metrics=["mrr", "recall@5"])
    # Relevant at rank 2 then rank 1: MRR = (0.5 + 1.0) / 2.
    assert result.metrics["mrr"] == pytest.approx(0.75)
    assert result.metrics["recall@5"] == pytest.approx(1.0)
    assert result.per_query["q1"]["mrr"] == pytest.approx(0.5)


def test_evaluate_engine_matches_on_chunk_id() -> None:
    rag = FakeRag({("q", "hybrid"): [hit(7, "somewhere.md")]})
    result = evaluate_engine(rag, [{"query": "q", "relevant": ["7"]}], metrics=["mrr"])
    assert result.metrics["mrr"] == pytest.approx(1.0)


def test_evaluate_engine_collapses_chunks_of_one_source() -> None:
    rag = FakeRag(
        {
            ("q", "hybrid"): [
                hit(1, "notes.md"),
                hit(2, "notes.md"),
                hit(3, "policy.md"),
            ]
        }
    )
    result = evaluate_engine(
        rag, [{"query": "q", "relevant": ["policy.md"]}], metrics=["mrr"]
    )
    # Three chunks, two sources: policy.md is the second distinct source.
    assert result.metrics["mrr"] == pytest.approx(0.5)


def test_evaluate_engine_supports_graded_relevance_and_ids() -> None:
    rag = FakeRag({("q", "hybrid"): [hit(1, "a.md"), hit(2, "b.md")]})
    dataset = [{"id": "graded", "query": "q", "relevant": {"a.md": 3, "b.md": 1}}]
    result = evaluate_engine(rag, dataset, metrics=["ndcg@2"])
    assert "graded" in result.per_query
    assert result.metrics["ndcg@2"] == pytest.approx(1.0)


def test_evaluate_engine_forwards_search_kwargs() -> None:
    rag = FakeRag({("q", "vector"): [hit(1, "a.md")]})
    result = evaluate_engine(
        rag,
        [{"query": "q", "relevant": ["a.md"]}],
        top_k=3,
        mode="vector",
        metrics=["mrr"],
    )
    assert rag.calls == [("q", 3, {"mode": "vector"})]
    assert result.metrics["mrr"] == pytest.approx(1.0)


def test_evaluate_engine_with_an_empty_dataset() -> None:
    result = evaluate_engine(FakeRag({}), [], metrics=["mrr"])
    assert result.metrics == {"mrr": 0.0}
    assert result.queries == 0


def test_evaluate_engine_rejects_malformed_entries() -> None:
    rag = FakeRag({})
    with pytest.raises(ConfigurationError):
        evaluate_engine(rag, [{"relevant": ["a.md"]}])
    with pytest.raises(ConfigurationError):
        evaluate_engine(rag, [{"query": "q"}])
    with pytest.raises(ConfigurationError):
        evaluate_engine(rag, ["not a mapping"])


def test_compare_runs_every_variant() -> None:
    rag = FakeRag(
        {
            ("q", "hybrid"): [hit(1, "a.md"), hit(2, "b.md")],
            ("q", "vector"): [hit(2, "b.md"), hit(1, "a.md")],
        }
    )
    dataset = [{"query": "q", "relevant": ["a.md"]}]
    results = compare(
        rag,
        dataset,
        {"hybrid": {"mode": "hybrid"}, "vector-only": {"mode": "vector"}},
        metrics=["mrr"],
    )
    assert list(results) == ["hybrid", "vector-only"]
    assert results["hybrid"].metrics["mrr"] == pytest.approx(1.0)
    assert results["vector-only"].metrics["mrr"] == pytest.approx(0.5)
    assert results["hybrid"].name == "hybrid"


def test_comparison_table_lines_up_variants() -> None:
    results = {
        "hybrid": EvalResult(metrics={"mrr": 1.0}, per_query={"q1": {"mrr": 1.0}}),
        "vector": EvalResult(metrics={"mrr": 0.5}, per_query={"q1": {"mrr": 0.5}}),
    }
    table = comparison_table(results)
    lines = table.splitlines()
    assert "hybrid" in lines[0] and "vector" in lines[0]
    assert "1.0000" in lines[2] and "0.5000" in lines[2]
    assert len(lines[0]) == len(lines[2])


def test_comparison_table_edge_cases() -> None:
    assert comparison_table({}) == "(no results)"
    assert comparison_table({"a": EvalResult()}) == "(no metrics)"
