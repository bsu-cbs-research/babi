# BABI Data Analysis

A desktop tool for the BABI lab that pairs infant breathing data with motion-capture data and flags the parts of the breathing trace where the signal looks unreliable, so researchers can focus their analysis on the time windows where the data is good.

---

## For Researchers (Non-Technical)

### What it does

Ingests raw capnostream and motion-capture files from a study session, aligns them on a common timeline, and flags the parts of the capnostream where the CO₂ waveform looks abnormal (e.g. sensor artefacts, dropouts) so you can exclude those moments from your analysis.

Each study session produces two kinds of recording:

- A **capnostream** file (`.xlsx`) — the breath/CO₂ waveform sampled 20 times per second.
- One or more **motion** files (`.tsv`) — body-marker positions sampled 100 times per second from the motion-capture rig.

1. Reads both kinds of file out of a folder you select.
2. Uses a small machine-learning model to scan the CO₂ waveform and mark each moment as either _normative_ (looks like a real breath) or _abnormal_ (sensor artefact, dropout, etc.).
3. Shifts the capnostream timeline by an offset you supply (in seconds) so it sits on top of the motion timeline.
4. Writes out three clean files you can drop straight into your analysis tools.

### Using the app

1. **Get the app.** Download the latest release for your platform from the [Releases](../../releases) page.
   - macOS: `babi-macos.tar.gz` — unzip and run `BABI Data Analysis.app`.
   - Windows: `babi-windows.zip` (folder bundle) or `babi-windows-onefile.zip` (single `.exe`).
2. **Prepare your data.** Put exactly one `.xlsx` capnostream file and one or more `.tsv` motion files into a single folder. Nothing else needs to be in the folder.
3. **Run the program.** Double-click the app icon. The window walks you through four screens:
   - **Start** — pick the folder.
   - **Configuration** — confirm the file count and enter the alignment offset (seconds; integers or decimals; positive shifts capnostream forward).
   - **Progress** — the app validates, parses, labels, and aligns. This typically takes a minute or two.
   - **Complete** — pick a destination folder. The app creates a sub-folder named `<source>_processed`.

### What you get back

Inside `<source>_processed/` you will find:

| File           | What it is                                                                                                                        |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `co2.csv`      | The full capnostream record with an extra `co2_valid` column (1 = normative, 0 = abnormal).                                       |
| `motion.csv`   | The combined motion record indexed by ISO-8601 timestamps, also carrying the `co2_valid` flag projected onto the motion timeline. |
| `metadata.txt` | The original metadata blocks lifted verbatim from each source file, for provenance.                                               |

If the source folder is malformed or the files don't pass validation, the app surfaces a specific error and lets you correct things without restarting.

---

## For Developers (Technical)

### Project layout

```
program/        Tkinter (ttkbootstrap) GUI: app.py + four screens + bundled static artifacts
pipeline/       Orchestrator: validate -> parse -> label -> align (used by GUI and CLI)
data/           Runtime parsing + validation; nested prepare/ tree drives training-data prep
labeling/       Runtime label() entry point, tflite inference (lite_runtime.py), and training
                code (autoencoder.py, processing.py)
alignment/      static() — re-indexes capnostream labels onto the motion DatetimeIndex
globals/        Shared constants (capnostream_sr=20, motion_sr=100)
notebooks/      Development notebooks: data.ipynb (data prep), labeling.ipynb (training)
scripts/        Build scripts + Keras→TFLite conversion
.github/        CI workflow that produces release artifacts
```

### Runtime pipeline

`pipeline.execute(folder, offset, status_cb=…, static_dir=…)` runs end-to-end:

1. **Discover** — scan the folder for exactly one `.xlsx` and one or more `.tsv` files.
2. **Validate** — `data.validating.capnostream` + `data.validating.motion` enforce the schema (header rows, marker names, sampling rates) so the parsers never see malformed input.
3. **Parse** — `data.parsing.capnostream` produces a 50 ms-indexed DataFrame; `data.parsing.motion` concatenates motion files, dedupes overlapping timestamps, and reindexes onto a 10 ms grid.
4. **Label** — `labeling.label()` slides a 20-sample window over the CO₂ waveform, runs the TFLite autoencoder, compares reconstruction error to the calibrated threshold, and assigns each sample a binary `co2_valid` label.
5. **Align** — `alignment.static` shifts capnostream by `offset` seconds and projects the labels onto the motion index with a 25 ms tolerance.

The GUI calls `pipeline.execute` from a worker thread (`program/screens/configuration.py`). Status messages are pushed onto a `queue.Queue` and pumped to the progress label by `BABIDataAnalysisApp.poll_queue` in `program/app.py:84`.

### Bundled artifacts

Three files in `program/static/` are read at runtime and shipped with every build:

| File            | Producer                                   | Consumer                           |
| --------------- | ------------------------------------------ | ---------------------------------- |
| `model.tflite`  | `scripts/convert_model_to_tflite.py`       | `labeling.lite_runtime.load_model` |
| `threshold.txt` | `labeling.autoencoder.calculate_threshold` | `labeling._load_artifacts`         |
| `scaling.txt`   | `labeling.processing.get_training_dataset` | `labeling._load_artifacts`         |

`labeling/__init__.py` looks them up under `static_dir` (defaulting to `program/static/`); the GUI passes a PyInstaller-aware `_MEIPASS` path so the same code works inside the packaged app.

### Development setup

Requires Python ≥ 3.12. The project uses [uv](https://github.com/astral-sh/uv):

```bash
uv sync                     # runtime deps only
uv sync --extra training    # adds keras + tensorflow for retraining
```

Run the app from source:

```bash
uv run python -m program.app
```

Run the pipeline headless against a sample folder (the `__main__` block in `pipeline/pipeline.py` defaults to `data/testing/p1`):

```bash
uv run python -m pipeline.pipeline
```

### Retraining the model

The trainer is split across two notebooks, both of which `cd ../` in cell 0 so paths are repo-relative.

1. **Prepare the dataset.** Drop raw recordings into `data/raw/` (paths in `data/constants.py`) and run `python -m data.scripts.parse` to materialise the per-baby pickles in `data/out/parsed/`. `notebooks/data.ipynb` exercises the extraction, augmentation, and pruning helpers under `data/prepare/`.
2. **Train.** Open `notebooks/labeling.ipynb` and run all cells:
   - Cell 2 builds the train/val/normative-test/abnormal-test splits and **writes `program/static/scaling.txt`**.
   - Cell 3 trains the convolutional autoencoder (`labeling.autoencoder.build`) with `EarlyStopping`, persisting the best weights to `program/static/model.keras`.
   - Cell 5 calls `calculate_threshold(..., export=True)`, which **writes `program/static/threshold.txt`**.
3. **Convert to TFLite.**

   ```bash
   uv run --extra training python scripts/convert_model_to_tflite.py
   ```

   This reads `program/static/model.keras` and writes `program/static/model.tflite`.

After step 3, `program/static/` contains the three artifacts the runtime needs and the next build will pick them up automatically. (`model.keras` is gitignored on purpose — only the TFLite artifact is tracked.)

### Building executables

A committed `program/static/model.tflite` is required before any build will run.

| Target                | Spec / command                                                         |
| --------------------- | ---------------------------------------------------------------------- |
| macOS `.app`          | `scripts/build_macos.sh` → `BABI Data Analysis.spec`                   |
| Windows folder bundle | `scripts/build_windows.sh` → `BABI Data Analysis.spec`                 |
| Windows single `.exe` | `scripts/build_windows_onefile.sh` → `BABI Data Analysis Onefile.spec` |

Both specs share the same `Analysis()` block (datas, hidden imports, exclusions); only the `EXE()` arguments differ. Keep them in sync when editing.

### Continuous integration

`.github/workflows/build.yml` runs on pushes to `main`, on `v*` tags, and on manual `workflow_dispatch`. The matrix builds:

- macOS: tarballs `BABI Data Analysis.app` → `babi-macos.tar.gz`.
- Windows: zips the folder bundle → `babi-windows.zip`, then runs the onefile spec → `babi-windows-onefile.zip`.

A version tag (`v0.1.2`) — or `workflow_dispatch` with `publish_release=true` and a `release_tag` — promotes the artifacts to a GitHub Release.
