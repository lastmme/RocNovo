# RocNovo — Quick start (Linux + conda)

This README describes how to set up a conda-based environment, prepare data(MGF -> HDF5), train CLIP (Stage One) and DeNovo (Stage Two) models, and run predictions. 


## 1. Create and activate a conda environment

Recommended: create a conda environment named `rocnovo` with Python 3.11:

```bash
conda create -n rocnovo python=3.11
conda activate rocnovo
```
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 2. Data preparation — convert MGF to HDF5

The repository contains a helper in `rocnovo/data/prepare_data.py`. The main convenience function is `MgfToHdf5(mgf_path, hdf5_path, **kwargs)`.

Example: convert a single MGF to HDF5 (from project root):

```python
from rocnovo.data.prepare_data import MgfToHdf5
MgfToHdf5('/data/raw/example.mgf', '/data/hdf5/example.hdf5', ms_level=2, annotated=False) 
# Convert MS2 spectra, unannotated
# or annotated=True if peptide labels are present. For training data, this is necessary.
# MgfToHdf5('/data/raw/example_annotated.mgf', '/data/hdf5/example_annotated.hdf5', ms_level=2, annotated=True)
```

- Place HDF5 paths in your YAML config under `trainer_data` / `predictor_data`.

## 3. Configuration files

Primary config files (root): `clip.yaml` and `denovo.yaml`. 
These YAML files contain inline comments describing each parameter.

Before training, update:
```yaml
trainer:
  max_epochs:  # total epochs to train
  warmup_epochs:  # fraction of epochs for LR warmup
  grad_norm_clip:  # gradient clipping norm
  model_save_folder: # directory to save model checkpoints
  validation_steps:  # validation frequency (steps)
  save_top_k:  # number of best checkpoints to keep
  show_progress_bar: # whether to show progress bar
  summarywriter: # TensorBoard log directory
  load_mode: # model loading mode. Options: "TFS", "RESUME", "WARM_START" 
# If training from scratch, set to "TFS".
#   "TFS": Basic Setup. Train model from scratch.
# If resuming interrupted training, set to "RESUME" or "WARM_START".
#   "RESUME": Restores the model weights, along with the optimizer and scheduler states.
#   "WARM_START": Restores only the model weights. This is equivalent to starting a new training session (from epoch 0) using pre-trained weights.


trainer_data:
  train_path: # path to training HDF5 file
  valid_path: # path to validation HDF5 file
  batch_size: # batch size for training
  data_loader_workers: # number of data loader workers
```

if needed, adapt model sizes, learning rates, etc. in the YAML.

Before inference, set:
```yaml
predictor_data:
  test_path: # path to test HDF5 file
  batch_size: # batch size for inference
  data_loader_workers: # number of data loader workers

# only needed for denovo prediction:
predictor:
  show_progress_bar: # whether to show progress bar
  beam_search:
    beam_size: # beam size for decoding
    max_len: # maximum peptide length        
    mass_tolerance: # mass tolerance (Da)
    max_isotope: # maximum isotope correction       
    min_peptide_len: # minimum peptide length
```


## 4. Train and predict
From the project root, run:

Train CLIP (Stage One):
```bash
python -m rocnovo.rocnovo clip train --config clip.yaml
``` 

Train DeNovo (Stage Two):
```bash
python -m rocnovo.rocnovo denovo train --config denovo.yaml 
```

Predict with DeNovo model:
```bash
python -m rocnovo.rocnovo denovo predict --config denovo.yaml --checkpoint /path/denovo.ckpt --output out_denovo.csv
```

## 5. Results

Predictions are written to CSV files specified by `--output`. Each row contains :
```python 
precursor_mz,precursor_charge,seq1_score,seq1,seq2_score,seq2,pred_mass,pred_score,pred_peptide,idx
```
where `seq1`/`seq2` are the predicted sequences from c-termini and n-termini, and `pred_peptide` is the final peptide selected from seq1 and seq2. `idx` is the index of the spectrum in the input HDF5 file.

## 6. Tips

- DDP (distributed data parallel) note — master address/port. This project sets the DDP master address and port in `rocnovo/common/ddp.py`:

  ```python
  def ddp_setup(rank: int, world_size: int):
      os.environ["MASTER_ADDR"] = "localhost"
      os.environ["MASTER_PORT"] = "12355"
      # rank identifies the logical GPU index
      init_process_group(backend="nccl", rank=rank, world_size=world_size)
  ```

  If port 12355 is already in use on your machine, distributed initialization may fail due to a port conflict. You can resolve this by editing `rocnovo/common/ddp.py` and changing "12355" to an unused port (for example "12356").





