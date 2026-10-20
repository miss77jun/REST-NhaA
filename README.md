# REST-NhaA

REST2 (Replica Exchange with Solute Tempering 2) enhanced sampling of the
NhaA-family membrane transporter 4AU5 (K300N mutant) in a CHARMM36m membrane
environment, built with OpenMM.

## Layout

- `4AU5/` — project working directory
  - `README.md` — full method description (standard REST2 + non-standard
    five-residue local-hot extension), CLI, results, and verification notes
  - `versions/rest2_4au5_v4.4.py` — current recommended script (v4.4)
  - `charmm-gui/4au5-aa-k300n/` — CHARMM-GUI build (PSF/PDB/toppar/OpenMM inputs)
  - `scripts/` — auxiliary analysis scripts
  - `environment_export.yml` — exported conda environment (OpenMM 8.5.1)

## Notes on this repository

Trajectories (DCD), restart files (RST), CHARMM run logs (\*.out), CUDA caches
and the raw CHARMM-GUI download package (`charmm-gui/charmm-gui-8425204688/`)
are intentionally not tracked — see `.gitignore`.

