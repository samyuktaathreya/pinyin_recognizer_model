from pathlib import Path

import pytest

import unicodedata

from pinyin_phoneme.labels import FINALS, INITIALS, split_syllable, standard_name, strip_tone_marks
from pinyin_phoneme.source_format import parse_path


@pytest.mark.parametrize("syllable, expected", [
    ("zhang", ("zh", "ang")),
    ("zhe", ("zh", "e")),
    ("an", ("none", "an")),
    ("er", ("none", "er")),
    ("a", ("none", "a")),
    ("shi", ("sh", "i")),
    ("xiong", ("x", "iong")),
    ("ju", ("j", "v")),
    ("que", ("q", "ve")),
    ("xuan", ("x", "van")),
    ("yun", ("y", "vn")),
    ("yi", ("y", "i")),
    ("wu", ("w", "u")),
    ("lu", ("l", "u")),
    ("lü", ("l", "v")),
    ("lv", ("l", "v")),
    ("nu:e", ("n", "ve")),
    ("guang", ("g", "uang")),
    ("ri", ("r", "i")),
    ("nen", ("n", "en")),
])
def test_split(syllable, expected):
    assert split_syllable(syllable) == expected


@pytest.mark.parametrize("bad", ["", "ng", "n", "hello", "zh", "xyz", "i", "u", "ong", "ia", "v"])
def test_split_rejects_non_pinyin(bad):
    with pytest.raises(ValueError):
        split_syllable(bad)


def test_label_lists_have_no_duplicates():
    assert len(set(INITIALS)) == len(INITIALS)
    assert len(set(FINALS)) == len(FINALS)


def test_tone_marks():
    assert strip_tone_marks("zhāng") == ("zhang", 1)
    assert strip_tone_marks("lǜ") == ("lv", 4)
    assert strip_tone_marks("ma") == ("ma", None)


def test_tone_marks_with_separate_accents():
    assert strip_tone_marks(unicodedata.normalize("NFD", "má")) == ("ma", 2)
    assert strip_tone_marks(unicodedata.normalize("NFD", "lǜ")) == ("lv", 4)


@pytest.mark.parametrize("given, expected", [
    ("ma", "ma"), ("lü", "lv"), ("lu:", "lv"), ("lv", "lv"), ("nüe", "nve"),
    ("lue", "lve"), ("nue", "nve"), ("ju", "ju"), ("jv", "ju"), ("yue", "yue"),
    ("xüan", "xuan"), ("ZHANG", "zhang"), ("lu", "lu"), ("nu", "nu"),
])
def test_standard_name(given, expected):
    assert standard_name(given) == expected


def test_parse_path_reads_the_standard_layout():
    assert parse_path(Path("uvm/zhang1.wav")) == ("uvm", "zhang", 1)
    assert parse_path(Path("misc/lv3.mp3")) == ("misc", "lv", 3)
    assert parse_path(Path("misc/ma5.mp3")) is None
    assert parse_path(Path("misc/ma.mp3")) is None
    assert parse_path(Path("left_out.csv")) is None
    assert parse_path(Path("a/b/ma1.mp3")) is None
