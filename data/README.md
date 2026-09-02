# Bundled data

`sample/` holds eight single-molecule trajectory tables produced by AAS, each
with the `hmm.csv` AAS itself wrote for it, and the raw image sequence each
table was derived from. They are what the figures quoted for this package were
measured on, so a reviewer can re-run the analysis and compare against AAS's
own result without obtaining anything else.

**For what these recordings are — the receptor, the stimulus, what the time
points correspond to — see the response letter accompanying the manuscript.**
This file describes only what the files themselves contain.

## The eight cells

Named as they are in SSBD. Two cells, four time points each.

| Cell | Time point | Trajectory table | AAS result | Image sequence |
|---|---|---|---|---|
| `egfr-EGF_t00002` | before stimulation | `egfr-EGF_t00002.csv` | `egfr-EGF_t00002_hmm.csv` | `egfr-EGF_t00002/` |
| `egfr-EGF_t00012` | 2 min | `egfr-EGF_t00012.csv` | `egfr-EGF_t00012_hmm.csv` | `egfr-EGF_t00012/` |
| `egfr-EGF_t00022` | 12 min | `egfr-EGF_t00022.csv` | `egfr-EGF_t00022_hmm.csv` | `egfr-EGF_t00022/` |
| `egfr-EGF_t00032` | 22 min | `egfr-EGF_t00032.csv` | `egfr-EGF_t00032_hmm.csv` | `egfr-EGF_t00032/` |
| `egfr-EGF_t000028` | before stimulation | `egfr-EGF_t000028.csv` | `egfr-EGF_t000028_hmm.csv` | `egfr-EGF_t000028/` |
| `egfr-EGF_t000128` | 2 min | `egfr-EGF_t000128.csv` | `egfr-EGF_t000128_hmm.csv` | `egfr-EGF_t000128/` |
| `egfr-EGF_t000228` | 12 min | `egfr-EGF_t000228.csv` | `egfr-EGF_t000228_hmm.csv` | `egfr-EGF_t000228/` |
| `egfr-EGF_t000328` | 22 min | `egfr-EGF_t000328.csv` | `egfr-EGF_t000328_hmm.csv` | `egfr-EGF_t000328/` |

The three files of a cell share its name, so the table, the result and the
images are matched by name alone.

Recorded 2022-11-02 between 18:41 and 19:15. The frame files inside each
sequence keep the names the acquisition software gave them, which is why they
do not match the cell name: those record which microscope run each frame came
from.

## File formats

| Kind | Description |
|---|---|
| `<cell>.csv` | Trajectory table. AAS v2 format: 18 columns, states named `Model 1`..`Model 5`, trajectory ends marked by an empty cell. |
| `<cell>_hmm.csv` | The VB-HMM result AAS produced for that table. The comparison target. |
| `<cell>/` | The recording, one TIFF per frame, 102 frames. |
| `settings.csv` | The AAS settings file for this session. See below. |

## Image sequences

816 files, 416 MB, 102 frames per recording with no missing frames. Two
numbering conventions appear, inherited from the acquisition software: one
recording numbers its frames `-10000` to `-10101`, the other seven `0000` to
`0101`. Nothing here depends on which is used.

**smDA-HMM does not read images.** It works from the trajectory tables only.
The sequences are included so the origin of those tables can be inspected;
viewing them needs an image program such as ImageJ.

They are also not in the packaged `.zip`, which would otherwise more than
quadruple in size for files the application cannot open. Clone the repository
to get them.

### Going back further than the tables

Reproducing a trajectory table from its image sequence is outside what this
package can do, and outside what can be fully checked from here: it takes
background subtraction in ImageJ followed by single-molecule tracking in AAS,
and AAS is shareware. What can be reproduced with what is provided is the step
from a trajectory table to the diffusion states — which is the part this
package implements.

## settings.csv

Included as a record of how AAS was configured for these recordings, so the
parameters can be checked against what this package uses.

**It is not read by the application.** smDA-HMM has no import-settings
feature; the values are entered in the interface, and it starts with them
already filled in. They agree exactly:

| Interface | Preset | settings.csv |
|---|---|---|
| Min States / Max States | 1 / 5 | `Minimum/Maximum number of states` 1 / 5 |
| Max Iterations | 100 | — (not recorded) |
| n_tilde | 1.0 | `n_tilde` 1 |
| c_tilde | 0.001 | `c_tilde` 0.001 |
| wPi_tilde | 1.0 | `w_pi_tilde` 1 |
| wB_tilde | 1.0 | `w_b_tilde` 1 |
| mag | 10.0 | `mag` 10 |
| Add prior per trajectory | on | `Add hyper params to each trajectory` YES |
| Calc KL per trajectory | on | `Calculate KL for each trajectory` YES |
| dt [s] | 0.040 | `Time per frame[ms]` 40 |
| um/px | 0.067 | `Distance per pixel[um]` 0.067 |

The remaining entries configure AAS's spot detection and tracking, which
produced these tables and is not part of this package. Among them are the
detection and linking settings: `Intensity Threshold`, `Connect distance`,
`Connect Frame`, `Minimum trajectory length`, `ROI` and `Scan Length`.
