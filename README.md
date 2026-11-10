# Deadline-constrained energy minimization via joint UAV trajectory and offloading design in UAV-aided MEC

Simulation code and raw results for the manuscript of the same title (author: Akash B, Rajalakshmi Engineering College).

## Contents
- `sim.py` - system models (free-space and line-of-sight channels; drag-only and complete rotary-wing power), the proposed BCD algorithm, the rate-bound SCA baseline, and the local, hover and fixed-circle baselines.
- `exp.py` - experiment runner (multiprocessing).
- `pos.py` - terminal-position-error study.
- `analyze.py` - statistics (paired savings, 95% intervals), figures and tables.
- `suite_*.json` - experiment definitions. `res_*.json` - raw results. `figures/` - figures of the paper.

## Reproduce
```
pip install -r requirements.txt
python3 exp.py suite_default.json res_default.json 200 2
python3 analyze.py
```
Arguments of `exp.py`: suite file, output file, number of placements, number of processes. Placements use seed 2026.

## Notes
- The rate-bound SCA baseline is our re-implementation of the technique used in Wu, Zeng and Zhang (IEEE TWC, 2018), not the original authors' code.
- Text and code were drafted with the help of an AI assistant (Claude, Anthropic), as declared in the manuscript.

## Citation
Please cite the manuscript (details to be added after publication). Archive DOI: add the Zenodo DOI here.

## Licence
MIT, see `LICENSE`.
