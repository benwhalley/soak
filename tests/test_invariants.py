"""Tests for soak.invariants -- declarative output checks on DAG nodes.

The invariant functions don't require a live DAG run; they take `(node, context)`
where `node` only needs the attributes the specific check touches (`.name`,
`.output`, `.inputs`, `.dag.nodes_dict`). We use lightweight stub objects
rather than spinning up a full pipeline.
"""

from types import SimpleNamespace

import pytest

from soak.error_handlers import NodeInvariantError
from soak.invariants import (
    INVARIANT_REGISTRY,
    extract_codes,
    extract_themes,
    invariant,
    run_invariants,
)
from soak.models.base import Code, Quote, Theme


# ---------- fixtures ---------------------------------------------------------


def _code(name: str = "Code name", description: str = "Code description text"):
    return Code(
        name=name,
        description=description,
        quotes=[Quote(text="example quote", source="docA__chunks__0")],
    )


def _theme(name: str, code_hashes=None):
    return Theme(
        name=name,
        description="A theme description longer than ten chars",
        code_hashes=list(code_hashes or []),
    )


def _node(name="themes", output=None, inputs=None, dag_nodes=None):
    dag = SimpleNamespace(nodes_dict=dag_nodes or {})
    return SimpleNamespace(name=name, output=output, inputs=inputs or [], dag=dag)


# ---------- registry --------------------------------------------------------


def test_registry_has_builtin_checks():
    assert "min_themes_with_codes" in INVARIANT_REGISTRY
    assert "code_retention_and_quotes" in INVARIANT_REGISTRY
    assert "non_empty_output" in INVARIANT_REGISTRY


def test_decorator_registers_new_check():
    @invariant("test_decorator_check")
    def _check(node, context):
        return None

    assert INVARIANT_REGISTRY["test_decorator_check"] is _check
    # cleanup so re-runs don't accumulate
    del INVARIANT_REGISTRY["test_decorator_check"]


def test_run_invariants_unknown_name_raises():
    node = _node()
    with pytest.raises(NodeInvariantError) as exc:
        run_invariants(node, ["does_not_exist"], {})
    assert "unknown invariant" in str(exc.value)


def test_run_invariants_wraps_unexpected_exception():
    @invariant("test_buggy_check")
    def _buggy(node, context):
        raise RuntimeError("oops")

    try:
        with pytest.raises(NodeInvariantError) as exc:
            run_invariants(_node(), ["test_buggy_check"], {})
        assert "RuntimeError" in str(exc.value)
    finally:
        del INVARIANT_REGISTRY["test_buggy_check"]


def test_run_invariants_no_op_when_empty():
    # None or [] = no checks, no raise
    run_invariants(_node(), [], {})
    run_invariants(_node(), None, {})


# ---------- min_themes_with_codes -------------------------------------------


def test_min_themes_passes_when_healthy():
    c = _code()
    themes = [_theme(f"Theme number {i:02d}", [c.hash()]) for i in range(5)]
    node = _node(output=themes)
    run_invariants(node, ["min_themes_with_codes"], {"min_themes": 3})


def test_min_themes_fails_when_too_few():
    c = _code()
    themes = [_theme("Single theme name", [c.hash()])]
    node = _node(output=themes)
    with pytest.raises(NodeInvariantError) as exc:
        run_invariants(node, ["min_themes_with_codes"], {"min_themes": 3})
    assert "only 1 themes" in str(exc.value)
    assert exc.value.invariant_name == "min_themes_with_codes"


def test_min_themes_fails_when_themes_have_no_codes():
    themes = [_theme(f"Theme number {i:02d}", []) for i in range(5)]
    node = _node(output=themes)
    with pytest.raises(NodeInvariantError) as exc:
        run_invariants(node, ["min_themes_with_codes"], {"min_themes": 3})
    assert "no linked codes" in str(exc.value)


def test_min_themes_uses_default_when_context_missing():
    # default is 3; 2 themes should fail
    c = _code()
    themes = [_theme(f"Theme number {i:02d}", [c.hash()]) for i in range(2)]
    with pytest.raises(NodeInvariantError):
        run_invariants(_node(output=themes), ["min_themes_with_codes"], {})


# ---------- code_retention_and_quotes ---------------------------------------


def test_code_retention_passes_when_healthy():
    input_codes = [_code(name=f"Code {i:02d}") for i in range(10)]
    output_codes = [_code(name=f"Out {i:02d}") for i in range(8)]

    upstream = SimpleNamespace(output=input_codes)
    node = _node(
        name="consolidated_codes",
        output=output_codes,
        inputs=["groupedcodes"],
        dag_nodes={"groupedcodes": upstream},
    )
    run_invariants(node, ["code_retention_and_quotes"], {})


def test_code_retention_fails_when_too_few():
    input_codes = [_code(name=f"Code {i:02d}") for i in range(100)]
    output_codes = [_code(name=f"Out {i:02d}") for i in range(5)]  # 5%, below 10%
    upstream = SimpleNamespace(output=input_codes)
    node = _node(
        name="consolidated_codes",
        output=output_codes,
        inputs=["groupedcodes"],
        dag_nodes={"groupedcodes": upstream},
    )
    with pytest.raises(NodeInvariantError) as exc:
        run_invariants(node, ["code_retention_and_quotes"], {})
    assert "retained only" in str(exc.value)


def test_code_retention_fails_when_codes_have_no_quotes():
    # construct a Code with empty quotes by bypassing validator
    c = _code()
    c.quotes = []  # post-hoc strip quotes
    c.resolved_quotes = None
    input_codes = [_code() for _ in range(3)]
    upstream = SimpleNamespace(output=input_codes)
    node = _node(
        name="consolidated_codes",
        output=[c, _code(), _code()],
        inputs=["groupedcodes"],
        dag_nodes={"groupedcodes": upstream},
    )
    with pytest.raises(NodeInvariantError) as exc:
        run_invariants(node, ["code_retention_and_quotes"], {})
    assert "no quotes" in str(exc.value)


def test_code_retention_skips_ratio_when_no_input():
    # if upstream has no codes, retention ratio check is skipped (still checks quotes)
    node = _node(
        name="consolidated_codes",
        output=[_code()],
        inputs=["groupedcodes"],
        dag_nodes={},  # upstream not in dag
    )
    # should not raise -- output codes have quotes, no ratio computed
    run_invariants(node, ["code_retention_and_quotes"], {})


# ---------- non_empty_output ------------------------------------------------


def test_non_empty_output_passes_when_populated():
    run_invariants(_node(output=[_code()]), ["non_empty_output"], {})
    run_invariants(_node(output={"x": 1}), ["non_empty_output"], {})


def test_non_empty_output_fails_on_none():
    with pytest.raises(NodeInvariantError):
        run_invariants(_node(output=None), ["non_empty_output"], {})


def test_non_empty_output_fails_on_empty_list():
    with pytest.raises(NodeInvariantError):
        run_invariants(_node(output=[]), ["non_empty_output"], {})


# ---------- DAGNode.validate_output dispatch ---------------------------------


def test_validate_output_calls_run_invariants(monkeypatch):
    """DAGNode.validate_output() delegates to run_invariants with self.context."""
    from soak.models.nodes import Map

    called = {}

    def fake_run(node, names, context):
        called["node"] = node
        called["names"] = names
        called["context"] = context

    monkeypatch.setattr("soak.invariants.run_invariants", fake_run)

    # construct a Map node with a stubbed context
    node = Map(name="x", inputs=["upstream"], template="t", invariants=["non_empty_output"])
    # bypass the heavy .context property
    type(node).context = property(lambda self: {"min_themes": 9})

    node.validate_output()

    assert called["names"] == ["non_empty_output"]
    assert called["context"] == {"min_themes": 9}
    assert called["node"] is node


def test_validate_output_no_op_when_invariants_unset():
    from soak.models.nodes import Map

    node = Map(name="x", inputs=["upstream"], template="t")
    # no invariants -> no exception even though we'd otherwise crash on .context
    node.validate_output()


# ---------- DAG / DAGNode exception API -------------------------------------


def test_dagnode_error_property_returns_underlying_error():
    """node.error is the public read of the internal _error attribute."""
    from soak.models.nodes import Map

    node = Map(name="x", inputs=["upstream"], template="t")
    assert node.error is None
    node._error = ValueError("boom")
    assert isinstance(node.error, ValueError)
    assert str(node.error) == "boom"


def test_dag_failed_node_finds_first_node_with_error():
    """DAG.failed_node returns the first declaration-order node with an error."""
    from soak.error_handlers import NodeInvariantError
    from soak.models.dag import DAG
    from soak.models.nodes import Map

    n1 = Map(name="a", inputs=["documents"], template="t")
    n2 = Map(name="b", inputs=["a"], template="t")
    n3 = Map(name="c", inputs=["b"], template="t")
    dag = DAG(name="test", nodes=[n1, n2, n3])

    assert dag.failed_node is None
    n2._error = NodeInvariantError("b", "broken")
    assert dag.failed_node is n2
    assert isinstance(dag.failed_node.error, NodeInvariantError)


def test_dag_execution_error_starts_as_none():
    """DAG.execution_error is initialized to None; populated by run() on failure."""
    from soak.models.dag import DAG
    from soak.models.nodes import Map

    dag = DAG(name="test", nodes=[Map(name="a", inputs=["documents"], template="t")])
    assert dag.execution_error is None
