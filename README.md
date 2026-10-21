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
- `00_verification/` — system correctness verification
  - `verify_rest2.py` — automated Layer 1 (static build) + Layer 2 (parameter/
    energy scaling) checks
  - `verification_plan_rest2_v44.md` — verification plan with feasibility
    assessment and missing-file inventory
- `environment_export.yml` mirror: see `4AU5/environment_export.yml` (OpenMM 8.5.1)

## Standard REST2 status

Standard REST2 and the local-hot extension share **one system build**
(`4AU5/charmm-gui/4au5-aa-k300n/`, 68,411 atoms — no second topology/box);
the difference lives only at the script-parameter level. Two paths run
standard REST2:

| Path | Location | Notes |
|---|---|---|
| `4AU5/versions/rest2_4au5_v4.3.py` | `local_heating_enabled: False` (default off) | pure standard REST2 script; produced the 50 ns control (`experiment_20260919_002619`, mean exchange 50.17%) |
| `4AU5/versions/rest2_4au5_v4.4.py --no_local_heating` | CLI flag | empties the hot-atom set, `lambda_hot=1.0`; mathematically equivalent to v4.3 with unified code — **recommended for future controls** |

Known gap: `4AU5/outputs/` is currently empty — the v4.3 50 ns control
products (DCD, checkpoints, `scientific_report.json`) were not included in
this deployment and exist only as historical records. Recover them from the
experimenter before reusing that baseline for trajectory-level comparisons.

## Notes on this repository

Trajectories (DCD), restart files (RST), CHARMM run logs (`*.out`), CUDA
caches, `.bak*` files and the raw CHARMM-GUI download package
(`charmm-gui/charmm-gui-8425204688/`) are intentionally not tracked — see
`.gitignore`.

