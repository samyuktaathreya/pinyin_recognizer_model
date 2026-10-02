"""Look through every recording and flag likely problems.

    python scripts/check_audio.py

Which folder gets checked is set by FOLDER_TO_CHECK just below the imports. It
starts as STANDARD_DIR (the output of standardize.py); change it to RAW_DIR to
look at the downloads as they came. Folder locations come from pinyin_phoneme/config.py.

Nothing is modified. Results go to <folder>/audio_report.csv (one row per file)
and a per-source summary is printed. Checks:

  unreadable          file is corrupt or not audio
  empty               almost no sound in it
  too_short           under 0.15 s of sound
  too_long            over 2.5 s of sound (several syllables? a whole sentence?)
  multiple_sounds     more than one separate stretch of sound
  quiet               loudest point is below -30 dBFS
  clipping            recorded too loud; the waveform is flattened at the top
  noisy               speech is less than 15 dB above the background
  cut_at_start/end    the sound is already loud at the very edge of the file,
                      so part of the syllable may be missing
  low_sample_rate     under 16 kHz; high frequencies (s, sh, x...) are lost
  dc_offset           waveform is not centred on zero (a recording fault)
  unrecognised_name   filename is not a pinyin syllable (+ optional tone)
  neutral_tone        named as tone 5 (or 0); these are left out of training

It also warns when two sources look like copies of the same recordings.

The noise figure is an estimate: it compares the loud parts of a file with its
quietest parts. A clip trimmed so tightly that it has no quiet part at all will
look noisier than it is; those usually also show cut_at_start or cut_at_end.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

from pinyin_phoneme.config import RAW_DIR, STANDARD_DIR  # noqa: F401
from pinyin_phoneme.labels import split_syllable, strip_tone_marks

# The folder to check: STANDARD_DIR or RAW_DIR.
FOLDER_TO_CHECK = STANDARD_DIR

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

# Thresholds. Adjust here if a check is too strict or too lenient.
MIN_SOUND_S = 0.15
MAX_SOUND_S = 2.5
QUIET_PEAK_DB = -30.0
CLIPPING_FRACTION = 0.001      # share of samples at full scale
NOISY_SNR_DB = 15.0
EDGE_DB = 10.0                 # edge counts as loud if within this of the speech level
MIN_SAMPLE_RATE = 16000
DC_OFFSET = 0.005
GAP_BETWEEN_SOUNDS_S = 0.35

FRAME_S = 0.01
_NAME = re.compile(r"([a-zü:]+)([0-5])?")


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def check_name(stem: str) -> str | None:
    plain, marked_tone = strip_tone_marks(stem.lower())
    match = _NAME.fullmatch(plain)
    if not match:
        return "unrecognised_name"
    try:
        split_syllable(match.group(1))
    except ValueError:
        return "unrecognised_name"
    tone = int(match.group(2)) if match.group(2) else marked_tone
    if tone in (0, 5):
        return "neutral_tone"
    return None


def analyse(path) -> dict:
    row = {"seconds": "", "sound_seconds": "", "sounds": "", "sample_rate": "",
           "channels": "", "peak_db": "", "snr_db": ""}
    issues = []
    name_issue = check_name(path.stem)
    if name_issue:
        issues.append(name_issue)

    try:
        wave, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        row["issues"] = " ".join(["unreadable"] + issues)
        return row

    row.update(sample_rate=rate, channels=wave.shape[1], seconds=round(len(wave) / rate, 2))
    wave = wave.mean(axis=1)
    offset = float(wave.mean()) if len(wave) else 0.0
    wave = wave - offset           # so an off-centre waveform does not skew the other checks
    frame = int(FRAME_S * rate)
    n = len(wave) // frame
    peak = float(np.abs(wave).max()) if len(wave) else 0.0
    if n < 5 or peak < 1e-4:
        row["issues"] = " ".join(["empty"] + issues)
        return row

    rms = np.sqrt((wave[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    smooth = uniform_filter1d(rms, 5)
    speech_level = float(np.percentile(smooth, 95))
    noise_level = float(np.percentile(smooth, 10))
    snr = min(db(speech_level) - db(noise_level), 60.0)

    # Which frames are sound: same rule as splitting.py.
    threshold = max(noise_level * 10 ** (10 / 20), smooth.max() * 10 ** (-35 / 20))
    loud = smooth > threshold
    sound_s = float(loud.sum() * FRAME_S)

    # Count separate stretches of sound.
    loud_frames = np.nonzero(loud)[0]
    sounds = 0
    if len(loud_frames):
        gaps = np.diff(loud_frames) * FRAME_S
        sounds = 1 + int((gaps >= GAP_BETWEEN_SOUNDS_S).sum())

    row.update(sound_seconds=round(sound_s, 2), sounds=sounds,
               peak_db=round(db(peak), 1), snr_db=round(snr, 1))

    if sound_s < 0.05:
        issues.append("empty")
    elif sound_s < MIN_SOUND_S:
        issues.append("too_short")
    if sound_s > MAX_SOUND_S:
        issues.append("too_long")
    if sounds > 1:
        issues.append("multiple_sounds")
    if db(peak) < QUIET_PEAK_DB:
        issues.append("quiet")
    if (np.abs(wave) >= 0.999).mean() > CLIPPING_FRACTION:
        issues.append("clipping")
    if snr < NOISY_SNR_DB:
        issues.append("noisy")
    edge = 3  # frames = 30 ms
    if db(float(rms[:edge].mean())) > db(speech_level) - EDGE_DB:
        issues.append("cut_at_start")
    if db(float(rms[-edge:].mean())) > db(speech_level) - EDGE_DB:
        issues.append("cut_at_end")
    if rate < MIN_SAMPLE_RATE:
        issues.append("low_sample_rate")
    if abs(offset) > DC_OFFSET:
        issues.append("dc_offset")

    row["issues"] = " ".join(issues)
    return row


def main() -> None:
    root = FOLDER_TO_CHECK
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in AUDIO_EXTENSIONS and ".git" not in p.parts)
    if not files:
        raise SystemExit(f"No audio files found under {root}")

    rows = []
    for i, path in enumerate(files, start=1):
        rel = path.relative_to(root)
        source = rel.parent.as_posix() or "."
        rows.append({"source": source, "file": path.name, **analyse(path)})
        if i % 2000 == 0:
            print(f"  {i}/{len(files)} files")

    report = root / "audio_report.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    def median(values):
        values = [v for v in values if v != ""]
        return float(np.median(values)) if values else float("nan")

    print(f"\n{len(files)} files in {len(by_source)} sources\n")
    for source, group in sorted(by_source.items()):
        rates = Counter(r["sample_rate"] for r in group if r["sample_rate"] != "")
        rate_text = ", ".join(f"{rate} Hz" for rate, _ in rates.most_common(2))
        print(f"{source}  ({len(group)} files, {rate_text})")
        print(f"    typical: {median(r['sound_seconds'] for r in group):.2f}s of sound, "
              f"peak {median(r['peak_db'] for r in group):.0f} dB, "
              f"speech {median(r['snr_db'] for r in group):.0f} dB above background")
        counts = Counter(issue for r in group for issue in r["issues"].split())
        if counts:
            print("    " + ", ".join(f"{name}: {count}" for name, count in counts.most_common()))
        else:
            print("    no issues found")

    # Do two sources hold the same recordings under different names? Websites copy
    # each other's audio, and one voice counted as two speakers would end up on
    # both sides of the train/test split. Same-named files that almost always
    # have the same length are a strong sign.
    lengths = {
        source: {r["file"].rsplit(".", 1)[0]: r["sound_seconds"]
                 for r in group if r["sound_seconds"] != ""}
        for source, group in by_source.items()
    }
    for a, b in combinations(sorted(lengths), 2):
        common = lengths[a].keys() & lengths[b].keys()
        if len(common) < 30:
            continue
        same = sum(abs(lengths[a][name] - lengths[b][name]) <= 0.02 for name in common)
        if same / len(common) >= 0.5:
            print(f"\nPOSSIBLY THE SAME RECORDINGS: {a} and {b} "
                  f"({same} of {len(common)} same-named files have matching lengths)")

    clean = sum(1 for r in rows if not r["issues"])
    print(f"\n{clean} of {len(rows)} files have no flags. Per-file details: {report}")


if __name__ == "__main__":
    main()
