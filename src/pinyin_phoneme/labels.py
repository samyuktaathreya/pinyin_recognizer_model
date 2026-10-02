"""What the model predicts, and how a pinyin syllable is split into those parts.

This file is the single source of truth for the label lists. Training, evaluation
and the exported labels.json all read from here.

Conventions:
- Labels follow pinyin spelling, so "y" and "w" count as initials.
- Syllables with no initial (a, er, ou...) get the initial "none".
- "v" stands for ü. After j, q, x and y, a written "u" is really ü, so it is
  stored as "v" (ju -> j + v, yuan -> y + van). That keeps each final label tied
  to one sound.
"""

from __future__ import annotations

import unicodedata

INITIALS = [
    "none",
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "zh", "ch", "sh", "r",
    "z", "c", "s",
    "y", "w",
]

FINALS = [
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "ong", "er",
    "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
    "ua", "uo", "uai", "ui", "uan", "un", "uang",
    "ve", "van", "vn",
]

INITIAL_TO_ID = {label: i for i, label in enumerate(INITIALS)}
FINAL_TO_ID = {label: i for i, label in enumerate(FINALS)}

# The only syllables with no initial. A bare "i", "u" or "ong" is not a syllable:
# those sounds are spelled yi, wu, weng.
ZERO_INITIAL_SYLLABLES = {"a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er", "o", "ou"}

_TWO_LETTER_INITIALS = ("zh", "ch", "sh")
_U_MEANS_V_AFTER = {"j", "q", "x", "y"}

# Tone-marked vowel -> (plain vowel, tone number)
_TONE_MARKS = {}
for _plain, _marked in {
    "a": "āáǎà", "e": "ēéěè", "i": "īíǐì",
    "o": "ōóǒò", "u": "ūúǔù", "v": "ǖǘǚǜ",
}.items():
    for _tone, _char in enumerate(_marked, start=1):
        _TONE_MARKS[_char] = (_plain, _tone)


def strip_tone_marks(text: str) -> tuple[str, int | None]:
    """'zhāng' -> ('zhang', 1). Returns tone None if there is no tone mark."""
    tone = None
    out = []
    # Some systems (macOS) store "á" as "a" + a separate accent; join them first.
    for char in unicodedata.normalize("NFC", text):
        if char in _TONE_MARKS:
            plain, tone = _TONE_MARKS[char]
            out.append(plain)
        else:
            out.append(char)
    return "".join(out), tone


def split_syllable(syllable: str) -> tuple[str, str]:
    """Split a toneless pinyin syllable into (initial, final).

    'zhang' -> ('zh', 'ang'),  'an' -> ('none', 'an'),  'ju' -> ('j', 'v')

    Raises ValueError for anything that is not a normal pinyin syllable.
    """
    s = syllable.strip().lower().replace("u:", "v").replace("ü", "v")

    if s[:2] in _TWO_LETTER_INITIALS:
        initial, final = s[:2], s[2:]
    elif s[:1] in INITIAL_TO_ID:
        initial, final = s[:1], s[1:]
    else:
        initial, final = "none", s

    if initial in _U_MEANS_V_AFTER and final.startswith("u"):
        final = "v" + final[1:]
    if final == "ue":  # lue / nue written without the umlaut
        final = "ve"

    if final not in FINAL_TO_ID or (initial == "none" and final not in ZERO_INITIAL_SYLLABLES):
        raise ValueError(f"not a pinyin syllable: {syllable!r}")
    return initial, final


def standard_name(syllable: str) -> str:
    """The one spelling used for filenames, matching the app.

    ü is written "v" where pinyin spells it ü (lv, nv, lve, nve), and
    ju / qu / xu / yu keep their "u".   'lü' -> 'lv',  'nue' -> 'nve',  'jv' -> 'ju'

    Raises ValueError if it is not a pinyin syllable.
    """
    s = syllable.strip().lower().replace("u:", "v").replace("ü", "v")
    if s[:1] in ("j", "q", "x", "y") and s[1:2] == "v":
        s = s[0] + "u" + s[2:]
    if s in ("lue", "nue"):
        s = s[0] + "ve"
    split_syllable(s)
    return s
