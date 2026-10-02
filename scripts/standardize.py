"""data/raw -> data/standard: one folder per speaker, every file named <syllable><tone>.

    python scripts/standardize.py

For each source listed in pinyin_phoneme/source_format.py this:
  - works out the syllable and tone from each filename
  - cuts four-tones-in-one-file recordings into four clips
  - leaves out neutral-tone files (tone 5, 0, or no tone given)
  - writes the result as data/standard/<speaker>/<syllable><tone>.<ext>

Spelling follows the app: ü is "v" in lv, nv, lve, nve; ju/qu/xu/yu keep "u".

data/raw is never modified. data/standard is deleted and rebuilt on every run.
Single-tone files are copied unchanged (same format); split clips are saved as wav.
Every file that was left out is listed, with the reason, in
data/standard/left_out.csv.
"""

from __future__ import annotations

import csv
import re
import shutil
import unicodedata
from collections import Counter, defaultdict

import soundfile as sf

from pinyin_phoneme.audio import AUDIO_EXTENSIONS
from pinyin_phoneme.config import RAW_DIR, STANDARD_DIR
from pinyin_phoneme.labels import split_syllable, standard_name, strip_tone_marks
from pinyin_phoneme.source_format import SOURCES
from pinyin_phoneme.splitting import split_four_tones

_NUMBERED = re.compile(r"(.+?)([0-5])")


# Syllables that are taught as a whole rather than spelled out as initial + final.
_READ_WHOLE = {"zhi", "chi", "shi", "ri", "zi", "ci", "si"}


def sounds_expected(syllable: str, intro: bool) -> set[int]:
    """How many stretches of sound a four-tone recording of this syllable may hold.

    4 = just the tones. 5 = the speaker spells the syllable out first.
    A speaker who spells things out only does so when there is an initial and a
    final to spell. For y-, w- and whole-read syllables it could go either way,
    so both are accepted.
    """
    if not intro:
        return {4}
    initial, _ = split_syllable(syllable)
    if initial == "none":
        return {4}
    if initial in ("y", "w") or syllable in _READ_WHOLE:
        return {4, 5}
    return {5}


class LeftOut(Exception):
    """Raised with the reason a file is not used."""


def syllable_and_tone(stem: str, naming: str) -> tuple[str, int | None]:
    """Work out (standard syllable, tone) from a filename without its extension.

    Tone is None for four_tones sources, where one file holds all four.
    """
    stem = unicodedata.normalize("NFC", stem).strip().lower()

    if naming == "numbered":
        match = _NUMBERED.fullmatch(stem)
        if not match:
            raise LeftOut("no tone number in name")
        syllable, tone = match.group(1), int(match.group(2))
    elif naming == "tone_marks":
        syllable, tone = strip_tone_marks(stem)
    elif naming == "tone1":
        syllable, tone = stem, 1
    elif naming == "four_tones":
        syllable, tone = stem, None
    else:
        raise ValueError(f"unknown naming style: {naming}")

    if naming != "four_tones" and tone not in (1, 2, 3, 4):
        raise LeftOut("neutral tone")
    try:
        return standard_name(syllable), tone
    except ValueError:
        raise LeftOut("not a pinyin syllable") from None


def main() -> None:
    if not RAW_DIR.is_dir():
        raise SystemExit(f"{RAW_DIR} does not exist")
    if STANDARD_DIR.exists():
        shutil.rmtree(STANDARD_DIR)
    STANDARD_DIR.mkdir(parents=True)

    kept = defaultdict(Counter)          # speaker -> tone -> files written
    left_out = []                        # (source folder, file, reason)
    missing_folders = []
    empty_folders = []                   # folder exists but has no audio files directly in it

    for source in SOURCES:
        folder = RAW_DIR / source.folder
        if not folder.is_dir():
            missing_folders.append(source.folder)
            continue
        out_dir = STANDARD_DIR / source.speaker
        out_dir.mkdir(exist_ok=True)

        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
        if not files:
            empty_folders.append(source.folder)
        for path in files:
            try:
                syllable, tone = syllable_and_tone(path.stem, source.naming)

                if source.naming == "four_tones":
                    try:
                        wave, rate = sf.read(str(path), dtype="float32", always_2d=True)
                    except Exception:
                        raise LeftOut("could not read file") from None
                    expected = sounds_expected(syllable, source.intro)
                    clips, sounds = split_four_tones(wave.mean(axis=1), rate, expected)
                    if clips is None:
                        where = " ".join(f"{s:.2f}-{e:.2f}" for s, e in sounds)
                        want = " or ".join(str(n) for n in sorted(expected))
                        raise LeftOut(f"could not split: found {len(sounds)} sounds, expected {want} ({where})")
                    targets = [out_dir / f"{syllable}{t}.wav" for t in (1, 2, 3, 4)]
                    if any(target.exists() for target in targets):
                        raise LeftOut("duplicate of a file already kept")
                    for t, (clip, target) in enumerate(zip(clips, targets), start=1):
                        sf.write(str(target), clip, rate, subtype="PCM_16")
                        kept[source.speaker][t] += 1
                else:
                    # Same speaker must not have ma1.mp3 and ma1.wav side by side.
                    if any(out_dir.glob(f"{syllable}{tone}.*")):
                        raise LeftOut("duplicate of a file already kept")
                    shutil.copyfile(path, out_dir / f"{syllable}{tone}{path.suffix.lower()}")
                    kept[source.speaker][tone] += 1
            except LeftOut as reason:
                left_out.append((source.folder, path.name, str(reason)))

    with (STANDARD_DIR / "left_out.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "file", "reason"])
        writer.writerows(left_out)

    # Report
    genders = {source.speaker: source.gender for source in SOURCES}
    print(f"{'speaker':16s} {'':6s} {'files':>6s}   tone 1 / 2 / 3 / 4")
    for speaker in sorted(kept):
        tones = kept[speaker]
        print(f"{speaker:16s} {genders[speaker]:6s} {sum(tones.values()):6d}   "
              + " / ".join(str(tones[t]) for t in (1, 2, 3, 4)))
    total = sum(sum(tones.values()) for tones in kept.values())
    print(f"\nWrote {total} files for {len(kept)} speakers to {STANDARD_DIR}")

    if left_out:
        print(f"\nLeft out {len(left_out)} files:")
        by_reason = Counter((folder, reason.split(":")[0]) for folder, _, reason in left_out)
        for (folder, reason), count in sorted(by_reason.items()):
            print(f"  {folder}: {count} x {reason}")
        print(f"Full list: {STANDARD_DIR / 'left_out.csv'}")
    for folder in missing_folders:
        print(f"\nWARNING: folder not found: {RAW_DIR / folder}")
    for folder in empty_folders:
        inside = sorted(p.name for p in (RAW_DIR / folder).iterdir())[:5]
        print(f"\nWARNING: no audio files directly inside {RAW_DIR / folder}")
        print(f"  it contains: {', '.join(inside) or 'nothing'}")
    if missing_folders or empty_folders:
        print("Check the folder names in pinyin_phoneme/source_format.py.")


if __name__ == "__main__":
    main()