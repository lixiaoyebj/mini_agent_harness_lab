import pytest
from tools import csv_statistics

def test_statistics():
    result =csv_statistics("sample.csv")
    assert result["rows"] == 5
    assert "pressure" in result["columns"]

    stats = result["statistics"]["pressure"]
    assert stats["max"] == pytest.approx(0.16)
    assert stats["mean"] == pytest.approx(0.132)


def test_path_traversal():
    with pytest.raises(ValueError):
        csv_statistics("../main.py")


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        csv_statistics("not_exist.csv")

