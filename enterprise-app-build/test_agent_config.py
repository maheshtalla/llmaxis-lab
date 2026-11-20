from __future__ import annotations

import ast
from pathlib import Path


AGENT_SOURCE = Path(__file__).with_name("agent.py")


def test_country_capital_schema_and_agent_configuration() -> None:
    tree = ast.parse(AGENT_SOURCE.read_text(encoding="utf-8"))
    schema = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CountryCapital"
    )
    assert any(
        isinstance(base, ast.Name) and base.id == "BaseModel"
        for base in schema.bases
    )
    capital = next(
        node
        for node in schema.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "capital"
    )
    assert isinstance(capital.annotation, ast.Name)
    assert capital.annotation.id == "str"

    agent_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Agent"
    )
    arguments = {keyword.arg: keyword.value for keyword in agent_call.keywords}

    assert isinstance(arguments["model"], ast.Constant)
    assert arguments["model"].value == "gemini-3.6-flash"
    assert isinstance(arguments["output_schema"], ast.Name)
    assert arguments["output_schema"].id == "CountryCapital"
    for transfer_option in (
        "disallow_transfer_to_parent",
        "disallow_transfer_to_peers",
    ):
        assert isinstance(arguments[transfer_option], ast.Constant)
        assert arguments[transfer_option].value is True
