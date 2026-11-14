# Convert MGF to HDF5

The converter is implemented in [`rocnovo/data/prepare_data.py`](../rocnovo/data/prepare_data.py). Use the Python function `MgfToHdf5`; running that module directly does not provide a conversion CLI.

## Convert an annotated test/training file

Run the following from the source repository root in the installed RocNovo environment:

```python
from pathlib import Path
from rocnovo.data.prepare_data import MgfToHdf5

source = Path("data/example.mgf")
output = Path("data/example.hdf5")
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise FileExistsError(output)  # The converter itself overwrites existing files.

MgfToHdf5(str(source), str(output), ms_level=2, annotated=True)
```

Each spectrum should provide a valid `PEPMASS`, `CHARGE`, peptide `SEQ`, and m/z–intensity peak pairs. Example:

```text
BEGIN IONS
TITLE=example_0
PEPMASS=500.25
CHARGE=2+
SEQ=PEPTLDE
100.0 10.0
200.0 20.0
END IONS
```

This block illustrates the file format only; it is not a mass-consistent evaluation example.

For unannotated inference inputs, call with `annotated=False`. For training or evaluating predictions against reference peptides, use `annotated=True` and ensure every accepted spectrum has a valid `SEQ` field.

## Parameters

| Argument | Default | Meaning |
| --- | --- | --- |
| `mgf_path` | Required | Input MGF file |
| `hdf5_path` | Required | Output HDF5 filename; existing file is replaced |
| `ms_level` | `2` | Controls precursor parsing; use 2 for the MS/MS workflow |
| `valid_charge` | Integers 1–10 | Accepted precursor charges; `None` disables this restriction |
| `annotated` | `False` | Read peptide strings from the MGF `SEQ` field |

`ms_level` is not an independent MGF spectrum-level filter: MGF parsing here uses it to select how precursor information is read. Missing/invalid precursor values or unsupported charge states can cause spectra to be skipped. A missing charge defaults to zero and is excluded by the default charge range.

## Peptide modification notation

This function copies peptide strings from `SEQ`; it does **not** automatically map source-specific modification names, normalize I/L, or validate that every peptide token is supported. Convert source modification notation to the target tokenizer vocabulary before running it. Inspect [`rocnovo/tokenizer/peptide.py`](../rocnovo/tokenizer/peptide.py) for supported tokens and tokenizer behavior. Format conversion alone is not normalization, deduplication, or a train/test overlap check.

## Output layout

```text
example.hdf5
└── 0/
    ├── metadata       # One structured row per accepted spectrum
    ├── spectra        # Concatenated peak arrays
    └── annotations    # Peptide labels when requested and available
```

| Object / field | Meaning |
| --- | --- |
| `0/metadata.precursor_mz` | Precursor m/z, float32 |
| `0/metadata.precursor_charge` | Charge, uint8 |
| `0/metadata.offset` | Starting peak offset, uint64 |
| `0/metadata.scan_id` | Original zero-based MGF record counter, uint32; not derived from TITLE |
| `0/spectra.mz_array` | Peak m/z, float64 |
| `0/spectra.intensity_array` | Peak intensity, float32 |
| `0/annotations` | UTF-8 peptide strings for annotated inputs |

File/group attributes include spectrum/peak counts. Group attribute `path` stores the supplied MGF path; use a non-private relative path if the resulting file will be shared. Peaks for sample i occupy `offset[i]:offset[i+1]`, or `offset[i]:n_peaks` for the last sample.

The converter stores the parsed peaks; model-specific peak selection and normalization are handled separately by the spectrum tokenizer.

## Check the converted file

```python
import h5py

with h5py.File("data/example.hdf5", "r") as f:
    group = f["0"]
    n = len(group["metadata"])
    print("spectra:", n, "peaks:", len(group["spectra"]))
    assert n > 0
    assert len(group["annotations"]) == n  # For annotated inputs only.
    print(group["annotations"].asstr()[:3])
```

The current parser appends an annotation before checking the charge filter. If an annotated spectrum is subsequently skipped, annotations and metadata can become misaligned. Prepare inputs with valid accepted charges and required fields, check skipped-spectrum warnings and annotation counts, and verify label ordering before use. A matching count alone does not prove correct ordering. Do not use `valid_charge=None` merely to bypass invalid precursor information.

The implementation collects peak arrays in memory, so large MGF files require sufficient RAM. An empty or fully rejected input cannot be concatenated successfully. Split inputs by a stable dataset/species boundary when appropriate, and preserve ordering for prediction/evaluation alignment.

## Use the file

Set `predictor_data.test_path` to the converted HDF5 file for inference, or `trainer_data.train_path` / `valid_path` for training. Configuration paths resolve from the working directory. See the [main README](../README.md) for model commands and the [dataset repository](https://modelscope.cn/datasets/lastmme/RocNovo-evaluation) for benchmark inputs and evaluation instructions.
