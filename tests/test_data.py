import pandas as pd
import pytest

from fintech_support.data import (
    ensure_known_labels,
    label_maps,
    read_split,
    training_split,
)


def test_split_preserves_classes_and_does_not_overlap():
    frame = pd.DataFrame(
        {
            "text": [f"example-{i}" for i in range(60)],
            "category": ["card"] * 30 + ["transfer"] * 30,
        }
    )
    train, validation = training_split(frame, validation_size=0.2)
    assert len(train) == 48
    assert len(validation) == 12
    assert set(train["text"]).isdisjoint(validation["text"])
    assert validation["category"].value_counts().to_dict() == {"card": 6, "transfer": 6}
    assert training_split(frame, validation_size=0.2)[1].equals(validation)


def test_label_map_is_stable_and_rejects_unseen_intents():
    train = pd.DataFrame({"category": ["transfer", "card", "transfer"]})
    label2id, id2label = label_maps(train)
    assert label2id == {"card": 0, "transfer": 1}
    assert id2label == {0: "card", 1: "transfer"}
    with pytest.raises(ValueError, match="Unknown intent labels"):
        ensure_known_labels(pd.DataFrame({"category": ["refund"]}), label2id)


def test_read_split_rejects_invalid_rows(tmp_path):
    (tmp_path / "train.csv").write_text("text,category\nquestion,card\n,transfer\n")
    with pytest.raises(ValueError, match="missing text"):
        read_split(tmp_path, "train")


def test_read_split_uses_only_expected_columns(tmp_path):
    (tmp_path / "test.csv").write_text("text,category,extra\n  hello  ,  card  ,foo\n")
    frame = read_split(tmp_path, "test")
    assert frame.to_dict("records") == [{"text": "hello", "category": "card"}]
