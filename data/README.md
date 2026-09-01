# Bundled data

`sample/` holds eight single-molecule trajectory tables produced by AAS, each
with the `hmm.csv` AAS itself wrote for it. They are what the figures quoted
for this package were measured on, so a reviewer can re-run the analysis and
compare against AAS's own result without obtaining anything else.

**For what these recordings are — the receptor, the stimulus, what the time
points correspond to — see the response letter accompanying the manuscript.**
This file describes only what the files themselves contain.

## Files

| Kind | Count | Description |
|---|---|---|
| `*_33fps.csv` | 8 | Trajectory tables. AAS v2 format: 18 columns, states named `Model 1`..`Model 5`, trajectory ends marked by an empty cell. |
| `*_33fps_hmm.csv` | 8 | The VB-HMM result AAS produced for the table beside it. The comparison target. |
| `settings.csv` | 1 | The AAS settings file for this session. See below. |

The eight are two series of four, distinguished in the file names:

```
RTK_B3_n0000_t0000_m00000_20221102_184109_L637-TIR-search_c1_v1.tif_SB25_BC0-1200_33fps.csv
        |    |     |      |
        |    |     |      +-- acquisition timestamp, 2022-11-02
        |    |     +--------- m00000 / p00004 / p00707 / p01315
        |    +--------------- t0000 .. t0003
        +-------------------- n0000 and n0009
```

Recorded 2022-11-02 between 18:41 and 19:15.

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

The remaining entries in `settings.csv` configure AAS's spot detection and
tracking, which produced these tables and is not part of this package.
