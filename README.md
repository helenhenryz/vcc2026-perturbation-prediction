# Predicting CRISPRi knockdowns in cell lines nobody has perturbed

Helen Henry (University of Tampere), Nyle Hussain (University of Glasgow), Kevin Henry.

Our final submission to the Arc Institute's Virtual Cell Challenge 2026, as one executed notebook and the package
behind it. The task: given only a human cell line's unperturbed control cells, predict 400 cells of whole-number RNA
counts (18,533 genes) for each of 300 CRISPRi knockdowns, in three lines no one has perturbed.

`notebook.ipynb` tells the story: the problem, the data, the pipeline built one measured piece at a time, and the final
file. Its code is the package `vccpipe/`, and together they are the pipeline that built the submission: run in `FULL`
mode, the notebook rebuilds the submitted file byte for byte. The outputs saved in the notebook show every figure and
number as that run computed them.

## The approach
- **Stage 1, the effect model.** For each knockdown, borrow the measured effect from public Perturb-seq screens of
  other cell lines: average them per cell type with inverse-noise weights, shrink noisy entries towards zero, and set
  each knockdown's size from how well the cell types agree on it and from statistics read from the line's own
  controls (its noise, and the power of the official statistical test).
- **Stage 2, the cell generator.** Draw 400 cells per knockdown from a model of the line's own controls (120 cell
  states, a negative binomial per gene), then fit the counts so their pooled profile is exactly stage 1's prediction.
- **Measured, not assumed.** A piece was kept only if it beat the noise floor on training lines held out like the
  challenge and scored by the official scorer. In the FULL run saved in the notebook, the final pipeline scores 0.391
  on six held-out lines, against 0.306 for the simple borrowed consensus and 0 for predicting no change (sections 8
  and 10).

## The repository
| path | what it is |
|---|---|
| `notebook.ipynb` | the story, executed, with every figure |
| `vccpipe/` | the pipeline and the notebook's figures, in thirteen modules (`inputs`, `stage1`, `generator`, `phase`, `harness`, `plots`, ...) |
| `tests/` | fast unit tests on toy data: no downloads, no GPU |
| `pyproject.toml` | the package, its pinned dependencies and the lint settings |
| `.github/workflows/ci.yml` | lint (ruff) and tests (pytest) on every push |

## Running it
- Python 3.12. Install the package and the notebook's tools at the pinned versions: `pip install -e ".[notebook]"`
  (add the `rebuild` extra for the appendix). Byte-for-byte reproduction also depends on numpy's bundled OpenBLAS;
  the notebook fixes the thread count of every step.
- Start Jupyter from the repository's root and run `notebook.ipynb`.
- `VCC_MODE=FAST` (the default) runs the exploration (sections 3 to 11) on a few thousand cells per training line,
  read from the public files by byte range, plus DepMap, the K562 genome-wide pseudobulk and the pathway screens
  (about 3.5 GB). The reads use at most four connections per server and back off whenever a server says it is busy;
  what was read is kept, so a rerun reads nothing twice. FAST never builds the final file.
- `VCC_MODE=FULL` runs the final build from the reference tables (a few hours, about 80 GB of memory at peak). With
  `VCC_REBUILD_TABLES=yes` the appendix first rebuilds every reference table from the public downloads and checks it
  against the shipped one (hours, several hundred GB of downloads).
- The sections that report the official scorer's numbers (2, 8, 10, 15) need a CUDA GPU: the scorer's DE engine needs
  one, and its CPU fallback gives different numbers. Sections 8 and 10 make about 90 official scorings (10 to 15 hours
  on one GPU); each score is saved under `data/scores` with the time it was made, and a rerun reads the saved scores
  instead of scoring again.
- Paths: `VCC_NB_DATA` (default `./data`), `VCC_TABLES`, `VCC_PHASE_DIR` (the challenge's final controls, downloaded
  with `vcc datasets download controls-final` on your own account), `VCC_WORK`.
- Nothing is submitted unless `VCC_SUBMIT=yes`.

## Tests
`pip install -e ".[dev]"`, then `ruff check . && ruff format --check . && pytest`. The tests build toy data in memory
and run in well under a minute on a CPU. The notebook's code is formatted for reading at 100 columns
(`ruff format --check --line-length 100 notebook.ipynb`), and a test checks that no two of its sections assign the
same name.

## Data and licences
No data is in this repository. The notebook downloads each dataset from its public source under that source's terms,
and checks every file by sha256. Our final submission uses X-Atlas/Orion, which is non-commercial (CC BY-NC-SA 4.0).

| data | source | licence as published |
|---|---|---|
| the challenge's control cells | Virtual Cell Challenge 2026 (`vcc datasets download`) | the challenge's terms; not redistributed |
| K562 essential, K562 genome-wide, RPE1 screens | Replogle et al. 2022, figshare+ | CC BY 4.0 |
| HepG2 and Jurkat screens | Nadig et al. 2025, GEO GSE264667 | public GEO deposit, no licence stated |
| H1 stem cells (2025 challenge) | Arc Institute, Virtual Cell Challenge 2025 and its Virtual Cell Atlas splits | public download, no licence stated |
| H1 genome-scale screen | Feng et al. 2025, GEO GSE295214 | MIT (the authors' records) |
| KOLF2.1J iPSC atlas | Nourreddine et al. 2026, figshare+ | CC BY 4.0 |
| CD4 T cells (pseudobulk) | Zhu, Dann et al. 2026 | MIT |
| pathway screens (six lines) | Jiang et al. 2025, Zenodo | CC BY 4.0 |
| K562 on 10x Flex (VIPerturb-seq) | Bradu et al. 2026, Zenodo | CC BY 4.0 |
| A549 CRISPRi | Myllia Biotechnology, GEO GSE337804 | public GEO deposit, no licence stated |
| A549 knockout atlas | Liu, Hillsley et al. 2026, GEO GSE345058 | public GEO deposit, no licence stated |
| A375 (PerturbFate) | GEO GSE291147 | public GEO deposit (data); the article is CC BY-NC-ND |
| DLD-1 | Genentech, GEO GSE337988 | public GEO deposit, no licence stated |
| HCT116 and HEK293T (X-Atlas/Orion) | Xaira Therapeutics | **CC BY-NC-SA 4.0** |
| cell-line expression catalogue | DepMap 24Q4, figshare+ | CC BY 4.0 |
| gene names | HGNC | CC0 |
| transcription start sites | GENCODE release 49 | EMBL-EBI terms: free use with attribution |

## Licence
The code is Apache-2.0 (`LICENSE`). Part of the cell generator is adapted from a public challenge entry under the MIT
licence; its notice is kept in `vccpipe/generator.py`, beside the code.
