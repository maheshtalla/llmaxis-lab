from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import main
    

def test_build_prompt_includes_masked_user_input() -> None:
    prompt = main.build_prompt("Contact [MASKED EMAIL]")

    assert "System Instructions:" in prompt
    assert "User Input: Contact [MASKED EMAIL]" in prompt
    assert "Assistant Response:" in prompt


def test_call_llm_is_explicit_about_mock_and_unimplemented_real_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "user_real_llm", False)
    assert isinstance(main.call_llm(main.build_prompt("hello")), str)

    monkeypatch.setattr(main, "user_real_llm", True)
    with pytest.raises(RuntimeError, match="Real LLM mode is not configured"):
        main.call_llm("hello")


def test_compliance_check_preserves_boolean_key_response_shape() -> None:
    assert main.compliance_check("This is safe.") == {False: "Response is compliant."}
    assert main.compliance_check("Do not disclose the password.") == {
        True: "Response contains blocked content."
    }


def test_process_request_sends_prompt_and_writes_only_masked_audit_data(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "audit_file", str(tmp_path / "audit_logs.json"))
    monkeypatch.setattr(main, "user_real_llm", False)
    prompts: list[str] = []

    def capture_prompt(prompt: str) -> str:
        prompts.append(prompt)
        return "This is a mock response from the AI model."

    monkeypatch.setattr(main, "call_llm", capture_prompt)
    client = TestClient(main.app)

    response = client.post(
        "/process_request",
        json={"user_input": "Contact alice@example.com for help."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["masked_input"] == "Contact [MASKED EMAIL] for help."
    assert payload["compliance_result"] == {"false": "Response is compliant."}
    assert len(prompts) == 1
    assert "Contact [MASKED EMAIL] for help." in prompts[0]
    assert "alice@example.com" not in prompts[0]
    audit = json.loads((tmp_path / "audit_logs.json").read_text(encoding="utf-8"))
    assert len(audit) == 1
    assert audit[0]["masked_input"] == payload["masked_input"]
    assert audit[0]["pii_types"] == ["email"]
    assert audit[0]["pii_counts"] == {"email": 1}
    assert "user_input" not in audit[0]
    assert "alice@example.com" not in json.dumps(audit)
    assert "pii_found" not in audit[0]


def test_compliance_typo_alias_remains_available() -> None:
    assert main.complaince_check("safe") == main.compliance_check("safe")


def test_root_route_lists_api_endpoints() -> None:
    response = TestClient(main.app).get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Enterprise AI Application API",
        "docs": "/docs",
        "health": "/health",
        "process_request": "/process_request",
    }


def test_audit_log_rewrites_valid_json_without_trailing_data(tmp_path, monkeypatch) -> None:
    audit_path = tmp_path / "audit_logs.json"
    monkeypatch.setattr(main, "audit_file", str(audit_path))
    audit_path.write_text(json.dumps([{"old": "a much longer existing record"}]), encoding="utf-8")

    main.write_audit_log("request-1", "safe", "ok", {})

    stored_text = audit_path.read_text(encoding="utf-8")
    assert json.loads(stored_text) == [
        {"old": "a much longer existing record"},
        {
            "request_id": "request-1",
            "masked_input": "safe",
            "response": "ok",
            "pii_types": [],
            "pii_counts": {},
        },
    ]
    assert not list(tmp_path.glob("tmp*"))
