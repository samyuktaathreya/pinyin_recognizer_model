import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "standardize", Path(__file__).parents[1] / "scripts" / "standardize.py")
standardize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standardize)


@pytest.mark.parametrize("stem, naming, expected", [
    ("a1", "numbered", ("a", 1)),
    ("lü4", "numbered", ("lv", 4)),
    ("Zhang3", "numbered", ("zhang", 3)),
    ("má", "tone_marks", ("ma", 2)),
    ("nüè", "tone_marks", ("nve", 4)),
    ("ai", "tone1", ("ai", 1)),
    ("bang", "four_tones", ("bang", None)),
])
def test_names_understood(stem, naming, expected):
    assert standardize.syllable_and_tone(stem, naming) == expected


@pytest.mark.parametrize("stem, naming, reason", [
    ("a5", "numbered", "neutral tone"),
    ("a0", "numbered", "neutral tone"),
    ("a", "numbered", "no tone number in name"),
    ("a", "tone_marks", "neutral tone"),
    ("ong", "tone1", "not a pinyin syllable"),
    ("readme1", "numbered", "not a pinyin syllable"),
])
def test_names_left_out(stem, naming, reason):
    with pytest.raises(standardize.LeftOut, match=reason):
        standardize.syllable_and_tone(stem, naming)


@pytest.mark.parametrize("syllable, intro, expected", [
    ("ban", True, {5}),        # spelled out: b, an, then four tones
    ("a", True, {4}),          # nothing to spell
    ("ai", True, {4}),
    ("er", True, {4}),
    ("yi", True, {4, 5}),      # could go either way
    ("wang", True, {4, 5}),
    ("shi", True, {4, 5}),
    ("shang", True, {5}),
    ("ban", False, {4}),       # a speaker who never spells
])
def test_sounds_expected(syllable, intro, expected):
    assert standardize.sounds_expected(syllable, intro) == expected


def test_same_recordings_share_a_speaker():
    from pinyin_phoneme.source_format import SOURCES
    speaker = {s.folder.split("/", 1)[1]: s.speaker for s in SOURCES}
    assert speaker["github-davinfifield_woman/audio"] == speaker["github-shikangkai_bothgender/audio-female"]
    assert speaker["gitcode.com_woman/audio"] == speaker["sreading.org_woman/audio"]
    assert speaker["lost-theory.org_man/audio"] == speaker["github-shikangkai_bothgender/audio-male"]
    assert len(set(speaker.values())) == 12