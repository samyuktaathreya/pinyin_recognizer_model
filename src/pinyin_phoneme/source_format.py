"""The ONE place that knows how the downloaded recordings are organised.

SOURCES lists every folder of recordings and how its files are named. The
folders all sit inside data/raw/<DOWNLOAD_FOLDER>/.
scripts/standardize.py reads this table and writes data/standard/, where every
speaker has one folder and every file is named <syllable><tone>, e.g. ma1.mp3.
Everything after that step only ever sees the standard layout.

To add a source, add a line to SOURCES. Naming styles:

    "numbered"    ma1.mp3, ma2.mp3 ...        tone is the digit
    "tone_marks"  mā.mp3, má.mp3 ...          tone is the accent
    "tone1"       ma.mp3                      every file is tone 1
    "four_tones"  ma.mp3 holding tones 1-4    cut into four clips
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Source:
    folder: str            # relative to data/raw/<DOWNLOAD_FOLDER>
    speaker: str           # folder name under data/standard; same name = same person
    gender: str            # "woman" or "man"
    naming: str = "numbered"
    intro: bool = False    # four_tones only: speaker spells most syllables out first


# The cloned repo inside data/raw that holds all the source folders below.
DOWNLOAD_FOLDER = "hanyu-pinyin-audio-data"

_SOURCES = [
    Source("digmandarin.com_woman/audio", "digmandarin", "woman"),
    Source("github-hotoo_woman/audio", "hotoo", "woman", naming="four_tones"),
    Source("github-hugolpz_woman/audio", "hugolpz", "woman"),
    Source("mandarintemple.com_woman/audio", "mandarintemple", "woman"),
    Source("studycli.org_woman/audio", "studycli", "woman"),
    Source("uvm.edu_man_four_tones_in_one_file/audio", "uvm", "man", naming="four_tones", intro=True),
    Source("yabla.com_woman/audio", "yabla", "woman"),
    Source("yoyochinese.com_woman/audio", "yoyochinese", "woman"),

    # Pairs of folders that hold the same recordings (confirmed by ear). Each pair
    # shares one speaker name, so the second folder only fills in syllables the
    # first one lacks. The folder listed first wins.
    Source("github-shikangkai_bothgender/audio-female", "davinfifield", "woman"),
    Source("github-davinfifield_woman/audio", "davinfifield", "woman"),

    Source("sreading.org_woman/audio", "sreading", "woman", naming="tone_marks"),
    Source("gitcode.com_woman/audio", "sreading", "woman"),

    Source("lost-theory.org_man/audio", "losttheory", "man"),
    Source("github-shikangkai_bothgender/audio-male", "losttheory", "man"),

    # The two misc folders are one speaker. The full set is listed first so that
    # it wins when both folders contain the same syllable.
    Source("misc_woman/2-所有音节", "misc", "woman"),
    Source("misc_woman/1-韵母", "misc", "woman", naming="tone1"),
]

# The same table with full paths relative to data/raw, which is what the scripts use.
SOURCES = [replace(s, folder=f"{DOWNLOAD_FOLDER}/{s.folder}") for s in _SOURCES]

SPEAKER_GENDER = {source.speaker: source.gender for source in SOURCES}

_STANDARD_NAME = re.compile(r"([a-z]+)([1-4])")


def parse_path(rel_path: Path) -> tuple[str, str, int | None] | None:
    """Read (speaker, syllable, tone) from a path inside data/standard.

    speaker/ma1.mp3 -> ("speaker", "ma", 1). Returns None for anything else.
    """
    if len(rel_path.parts) != 2:
        return None
    match = _STANDARD_NAME.fullmatch(rel_path.stem)
    if not match:
        return None
    return rel_path.parts[0], match.group(1), int(match.group(2))