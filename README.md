# pinyin-phoneme

A small model that listens to one spoken pinyin syllable and scores its initial
and its final. Trains on a laptop and exports to ONNX for use in the browser.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Steps

Put the downloaded recordings under `data/raw/`, then:

```
python scripts/standardize.py                            # raw -> data/standard/<speaker>/ma1.mp3
python scripts/check_audio.py                            # report problems in data/standard
python scripts/prepare_data.py                           # standard -> processed clips + splits
python -m pinyin_phoneme.train --name baseline           # -> runs/<date>_baseline/
python -m pinyin_phoneme.evaluate --run runs/<date>_baseline
python -m pinyin_phoneme.export --run runs/<date>_baseline    # -> exports/
```

Each step reads the previous step's output from disk, so any step can be rerun
on its own.

## The one file that knows about your downloads

`src/pinyin_phoneme/source_format.py` lists every folder under `data/raw`, who
the speaker is, and how its files are named (tone number, tone mark, tone 1
only, or four tones in one file). `standardize.py` uses that table to build
`data/standard`, where every file is `<speaker>/<syllable><tone>`. Files it
leaves out are listed with the reason in `data/standard/left_out.csv`.

Naming follows the app: ü is `v` in `lv`, `nv`, `lve`, `nve`; `ju`/`qu`/`xu`/`yu`
keep `u`. Neutral tone is not used.

## Layout

```
configs/baseline.yaml          every setting for a run
scripts/standardize.py         sort the downloads into one naming scheme
scripts/check_audio.py         flag quiet, clipped, cut-off or copied recordings
scripts/prepare_data.py        cleaning, clip-length measurement, speaker splits
scripts/make_fake_data.py      synthetic audio for testing the pipeline
src/pinyin_phoneme/
  config.py                    folder locations (RAW_DIR, RUNS_DIR, ...)
  source_format.py             the table of download folders  (edit this)
  splitting.py                 cuts four-tone recordings into four clips
  labels.py                    the initial and final lists; syllable splitting
  audio.py                     loading, resampling, silence trimming
  dataset.py                   clips + augmentation for training
  model.py                     spectrogram step + CNN with two heads
  train.py / evaluate.py / export.py
data/  runs/  exports/         generated, not committed
```

## Using the exported model

`exports/pinyin_phoneme.onnx` takes raw audio and returns probabilities:

| name | shape | meaning |
|---|---|---|
| `audio` (input) | `[batch, num_samples]` float32 | 16 kHz mono, values in -1..1 |
| `initial_probs` | `[batch, 24]` | one probability per entry of `initials` in labels.json |
| `final_probs` | `[batch, 35]` | one probability per entry of `finals` |

`exports/labels.json` holds `sample_rate`, `num_samples` and the two label
lists. The app must send exactly `num_samples` samples, padding with zeros or
cropping around the syllable.

To score an attempt at a known target, split the target the same way
`labels.split_syllable` does and read off the probability of its initial and
of its final.
