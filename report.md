# BABI Data Analysis Project Report

## Executive Overview

BABI Data Analysis is a desktop application and processing pipeline for pairing infant capnostream CO2 recordings with motion-capture recordings, automatically identifying unreliable regions of the CO2 waveform, and exporting analysis-ready datasets. The project is currently implemented as a Python 3.12 application with a Tkinter/ttkbootstrap graphical interface, a reusable headless pipeline, and packaged model artifacts for runtime anomaly detection.

The system addresses a practical research bottleneck: raw capnostream traces can contain sensor dropouts, artefacts, and other abnormal waveform regions that must be excluded or treated cautiously during downstream analysis. Instead of requiring manual review of the entire breathing trace, the software labels each CO2 sample as valid or invalid and projects that validity signal onto the higher-frequency motion-capture timeline.

Current status: the core workflow is functional and packaged for desktop use. The repository contains the runtime model artifact (`program/static/model.tflite`), threshold calibration (`program/static/threshold.txt`), scaling parameters (`program/static/scaling.txt`), application source code, build scripts, and CI configuration for macOS and Windows release artifacts. The main outstanding gap is formal validation reporting: the repository does not currently include a test suite or persisted model-performance metrics such as precision, recall, false-positive rate, or inter-rater comparison against expert labels.

## Research And Technical Objective

The project objective is to produce synchronized, quality-annotated datasets from study-session recordings. Each session is expected to contain exactly one capnostream Excel file and one or more motion-capture TSV files. The output is designed to support researchers who need clean temporal windows where breathing data can be trusted and aligned with infant movement.

The system produces three primary outputs:

- `co2.csv`: full capnostream record with an added `co2_valid` binary column.
- `motion.csv`: composite motion-capture record indexed by ISO-8601 timestamps with the CO2 validity label projected onto the motion timeline.
- `metadata.txt`: provenance record containing source metadata extracted from the raw capnostream and motion files.

## Methods

### Data Inputs

The pipeline assumes two acquisition modalities:

- Capnostream data are supplied as `.xlsx` files. The parser reads the instrument metadata, extracts a report-generation timestamp as the start time, replaces missing sentinel values, and builds a 50 ms sampling grid corresponding to 20 Hz CO2 waveform sampling.
- Motion-capture data are supplied as `.tsv` files. The parser reads motion metadata, expects 100 Hz acquisition, extracts marker columns, masks zero-valued marker measurements as missing data, concatenates multiple motion files, removes duplicate timestamps by averaging, and reindexes the composite frame onto a 10 ms grid.

### Validation And Parsing

Before parsing, the pipeline performs schema validation so malformed source folders fail early with actionable errors. Capnostream validation checks file existence, extension, minimum shape, required headers (`Date`, `Time`, `CO₂ Wave`), and expected timestamp formatting. Motion validation checks header keys, timestamp formatting, sampling frequency, non-empty marker names, the `Frame Time` data header, and marker-schema consistency across all motion files in the selected folder.

This validation layer is important for research operations because it makes failures attributable to specific data-layout problems rather than downstream numerical errors. It also constrains the system to known acquisition formats, improving reproducibility across sessions.

### CO2 Waveform Labeling

The labeling method uses an autoencoder-based anomaly-detection approach. The model is trained on normative CO2 waveform windows and then used to reconstruct incoming waveform windows. Reconstruction error is compared with a calibrated threshold; windows below the threshold are treated as normative and windows above the threshold are treated as abnormal.

The implemented training architecture is a compact one-dimensional convolutional autoencoder:

- Input: 20-sample waveform windows, corresponding to approximately one second at 20 Hz.
- Encoder: two Conv1D/MaxPooling stages followed by flattening.
- Bottleneck: dense latent representation of size 8.
- Decoder: dense expansion, reshape, upsampling, and Conv1D reconstruction layers.
- Loss: mean squared reconstruction error.

At runtime, the Keras model is converted to TensorFlow Lite and executed through `ai-edge-litert` or `tflite-runtime`, depending on platform availability. The current bundled model artifact is `program/static/model.tflite` and is approximately 19 KB. Runtime calibration currently uses:

- Reconstruction-error threshold: `0.001430183254159612`.
- Training-data scaling range: minimum `0.01`, maximum `34.897894736842105`.

The labeling implementation converts the raw CO2 signal into overlapping 20-sample windows, scales each window using the stored training range, evaluates all windows against the TFLite model, and expands normative window predictions back to sample-level `co2_valid` labels. This produces a binary mask over the original capnostream timeline.

### Temporal Alignment

The capnostream and motion streams are aligned by applying a user-supplied offset, in seconds, to the capnostream timeline. The shifted capnostream validity labels are then projected onto the motion-capture `DatetimeIndex` using nearest-neighbor reindexing with a 25 ms tolerance. Motion samples without a nearby capnostream label are marked invalid (`0`).

This design explicitly separates acquisition-time correction from signal classification. Researchers can adjust the offset while preserving a deterministic labeling and alignment procedure.

### Application Workflow

The graphical application exposes the pipeline through four screens:

- Start: folder selection.
- Configuration: file-count summary and alignment-offset entry.
- Progress: worker-thread execution with status updates for discovery, validation, parsing, labeling, and alignment.
- Complete: export of `co2.csv`, `motion.csv`, and `metadata.txt` to a `<source>_processed` folder.

The GUI runs the pipeline on a background thread so the interface remains responsive, and it passes a PyInstaller-aware static-resource path so the model artifacts can be found both during source execution and inside packaged applications.

## Results To Date

The implemented system currently demonstrates the following results:

- End-to-end data-processing path exists from raw session folder to synchronized output CSVs.
- CO2 validity labeling is integrated into the production pipeline rather than isolated in notebooks.
- Runtime inference no longer depends on TensorFlow/Keras; packaged use relies on a small TFLite artifact and lightweight runtime.
- Motion outputs carry the CO2 validity signal on the motion-capture timeline, allowing movement analyses to filter or stratify by respiratory signal quality.
- Source metadata are preserved in a separate text file, supporting provenance and auditability.
- Build automation exists for macOS and Windows artifacts, including Windows folder-bundle and single-executable packaging.

The repository does not currently provide quantitative model-evaluation results. Specifically, there are no committed artifacts reporting sensitivity, specificity, precision, recall, F1 score, ROC/PR curves, per-participant generalization, or comparison with human-reviewed labels. Therefore, the current result should be interpreted as an operational prototype with an integrated anomaly detector, not yet as a fully validated research instrument.

## Current Project Status

### Implemented

- Desktop GUI for researcher-facing use.
- Reusable pipeline entry point for headless execution.
- Structural validation for expected capnostream and motion file formats.
- Capnostream parser with 20 Hz synthetic timestamping.
- Motion parser with multi-file concatenation, duplicate-time handling, and 100 Hz reindexing.
- Autoencoder-based CO2 anomaly labeling using TFLite runtime inference.
- Static alignment of CO2 validity labels onto the motion timeline.
- Export workflow for processed CO2, processed motion, and source metadata.
- PyInstaller build scripts and GitHub Actions workflow for release artifacts.

### Evidence In Repository

- Runtime model artifacts are present under `program/static/`.
- CI checks that `program/static/model.tflite` exists before building desktop packages.
- Training and conversion workflow is documented in `README.md` and supported by notebooks/scripts.
- The codebase separates validation, parsing, labeling, alignment, GUI orchestration, and build automation into distinct modules.

### Not Yet Demonstrated In Repository

- Automated regression tests for parsing, validation, alignment, or labeling.
- A committed benchmark dataset or gold-standard label set for repeatable evaluation.
- Formal model-performance report across participants, sessions, sensor conditions, and artefact types.
- Release notes or acceptance criteria tying model versions to validation outcomes.
- Robustness evaluation for edge cases such as long gaps, extreme offset values, unexpected capnostream headers, or marker dropout patterns beyond zero masking.

## Risks And Limitations

The largest scientific risk is that the binary `co2_valid` label may be operationally useful but not yet empirically characterized. Without validation against expert-reviewed data, downstream analyses could either retain artefactual regions or discard valid respiratory signal. This is particularly important if excluded windows correlate with infant motion, sensor placement, or session condition.

The largest engineering risk is the absence of automated tests around file-format assumptions. The project validates expected headers and sampling rates, but capnostream and motion-capture exports can vary by software version, acquisition settings, or manual file handling. Small format changes could break parsing or silently alter alignment if not covered by fixtures.

The main reproducibility risk is model provenance. The deployed TFLite model, threshold, and scaling values are present, but the repository does not currently bind them to a documented training run, dataset snapshot, or evaluation report. For research governance, model artifacts should be versioned together with training data definitions and validation outcomes.

## Recommended Next Steps

1. Establish a gold-standard validation set with expert-reviewed normative and abnormal CO2 regions, ideally stratified by participant/session and artefact type.
2. Generate a model-validation report including confusion matrices, precision/recall, sensitivity/specificity, false-positive and false-negative examples, and performance by session.
3. Add automated tests with small representative capnostream and motion fixtures covering validation failures, parser outputs, label length preservation, and alignment tolerance behavior.
4. Version model artifacts with provenance metadata: training date, dataset identifier, code commit, threshold percentile, scaling range, and validation metrics.
5. Add a lightweight quality-control summary to exported outputs, such as total recording duration, percent valid CO2, number and duration of invalid segments, and motion samples without nearby CO2 labels.

## General Status Summary

BABI Data Analysis is in a strong applied-prototype state. The core research workflow is implemented, the interface is usable by non-technical researchers, and packaging infrastructure is in place for distribution. The technical architecture is appropriate for the task: validation isolates file-format issues, the model is lightweight enough for desktop deployment, and alignment is explicit and reproducible.

Before the system is treated as a validated research-analysis instrument, the project should prioritize empirical model validation, regression testing, and model-provenance documentation. These additions would move the work from a functional processing tool to a defensible research platform suitable for director-level review, cross-session deployment, and downstream scientific interpretation.
