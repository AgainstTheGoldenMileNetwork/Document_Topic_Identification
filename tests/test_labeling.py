import json
from types import SimpleNamespace

from labeling import build_prompt, generate_labels

TOPICS = {
    0: {"top_terms": ["patent", "invention"], "representative_docs": ["a"]},
    1: {"top_terms": ["vessel", "cargo"], "representative_docs": ["b"]},
}
DOCS = {"a": {"syllabus": "Patent for a saw guide."}, "b": {"text": "The ship's cargo was lost."}}


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def fake_client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def reply(payload, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=json.dumps(payload))])


def test_prompt_includes_keywords_and_excerpts():
    prompt = build_prompt(TOPICS, DOCS)
    assert "patent, invention" in prompt and "saw guide" in prompt and "cargo was lost" in prompt


def test_labels_are_parsed_and_cached(tmp_path):
    messages = FakeMessages(reply({"labels": [
        {"topic_id": 0, "label": "Patent law"}, {"topic_id": 1, "label": " Admiralty cargo claims "},
        {"topic_id": 9, "label": "ignored"},
    ]}))
    labels = generate_labels(TOPICS, DOCS, "claude-opus-5-5", tmp_path, client=fake_client(messages))
    assert labels == {0: "Patent law", 1: "Admiralty cargo claims"}
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"

    # Second call hits the cache, not the API.
    again = generate_labels(TOPICS, DOCS, "claude-opus-5-5", tmp_path, client=fake_client(FakeMessages(error=AssertionError)))
    assert again == labels


def test_failures_fall_back_to_empty(tmp_path):
    assert generate_labels(TOPICS, DOCS, "m", tmp_path, client=fake_client(FakeMessages(error=RuntimeError("boom")))) == {}
    refused = FakeMessages(reply({"labels": []}, stop_reason="refusal"))
    assert generate_labels(TOPICS, DOCS, "m", tmp_path, client=fake_client(refused)) == {}
    garbled = FakeMessages(SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="not json")]))
    assert generate_labels(TOPICS, DOCS, "m", tmp_path, client=fake_client(garbled)) == {}
    assert not list(tmp_path.glob("labels-*.json"))
