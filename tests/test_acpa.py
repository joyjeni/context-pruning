from acpa_gemma.acpa import AdaptiveContextPruner, ContextElement


def test_dependency_context_is_preserved_when_pruning():
    context = [
        ContextElement(id="a", text="tool step 1 shows prompt injection", importance_score=0.7),
        ContextElement(id="b", text="irrelevant filler", importance_score=0.1),
        ContextElement(id="c", text="more irrelevant filler", importance_score=0.1),
    ]
    pruner = AdaptiveContextPruner(prune_ratio=0.67)

    retained, stats = pruner.competitive_eviction(
        context,
        current_timestamp=1,
        citations={"tool step 1"},
    )

    assert "a" in {element.id for element in retained}
    assert stats["n_dependencies_preserved"] == 1
    assert stats["n_evicted"] >= 1


def test_frequently_accessed_context_gets_priority_boost():
    context = [
        ContextElement(id="hot", text="credential leak evidence", importance_score=0.2),
        ContextElement(id="cold", text="background note", importance_score=0.9),
    ]
    pruner = AdaptiveContextPruner(
        alpha=3.0,
        beta=0.1,
        gamma=0.0,
        prune_ratio=0.5,
        cache_threshold=1,
    )
    pruner.track_context_access(context, "agent", "credential leak should be blocked")

    retained, _ = pruner.competitive_eviction(context, current_timestamp=1)

    assert [element.id for element in retained] == ["hot"]
    assert context[0].cache_priority > 1.0
