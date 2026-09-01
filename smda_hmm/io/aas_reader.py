"""AAS v2/v4 readers: data.csv into a normalised frame, hmm.csv into a dict.

Ported from smDA-Python (smda/io/aas2_reader.py).  detect_aas_version is
not carried over -- version detection lives in aas_format, which is the
one place that knows how the formats differ.

Original docstring:
AAS2 format reader (data.csv, hmm.csv, settings.csv).

AAS2 data.csv has 18 columns with `Model 1-5` (state assignments per K-model)
and `Label` column. AAS4 has 19 columns with `state(diffusion) 1-5`, `MSE`,
and `Contours [json]`.

This module provides:
- load_aas2_data_csv(): Read AAS2 data.csv → standardized DataFrame
- load_aas2_hmm_csv(): Parse AAS2 hmm.csv → structured dict
- load_aas_settings_csv(): Parse AAS settings.csv → param dict
- (version detection lives in aas_format.detect_version)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# AAS version detection
# ============================================================================


# ============================================================================
# AAS2 data.csv reader
# ============================================================================

def load_aas2_data_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read AAS2 data.csv and return standardized DataFrame.

    AAS2 columns (18):
        No, Time [frame], xg [px], yg [px], sigma x [px], sigma y [px],
        Imax [a.u.], Iback [a.u.], a [a.u./px], b[a.u./px],
        Raw Intensity[au], Average Intensity[au/px^2],
        Model 1, Model 2, Model 3, Model 4, Model 5, Label

    Returns DataFrame with standardized column names matching AAS4:
        No, Time [frame], xg [px], yg [px], sigma x 1 [px], sigma y 1 [px],
        Imax 1 [a.u.], Iback 1 [a.u.], a 1 [a.u./px], b 1 [a.u./px],
        Raw Intensity 1 [a.u.], Average Intensity 1 [a.u./px^2],
        state(diffusion) 1-5, Label
    """
    df = pd.read_csv(csv_path)

    # Rename AAS2 columns to standardized names
    rename_map = {}
    for col in df.columns:
        # Model N → state(diffusion) N
        m = re.match(r'Model\s+(\d+)', col)
        if m:
            rename_map[col] = f'state(diffusion) {m.group(1)}'

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# ============================================================================
# AAS2 hmm.csv reader
# ============================================================================

def load_aas2_hmm_csv(hmm_path: str | Path) -> dict:
    """Parse AAS2 hmm.csv and return structured result.

    Returns:
        {
            'best_n': int,
            'method': str,
            'models': {
                K: {
                    'lower_bound': float,
                    'lnZs': float,
                    'kl': float,
                    'kl_pi': float,
                    'kl_diffusion': float,
                    'kl_b': float,
                    'D_mean': ndarray (K,),   # μm²/s
                    'D_std': ndarray (K,),
                    'pi_mean': ndarray (K,),
                    'pi_std': ndarray (K,),
                    'A_mean': ndarray (K,K),  # transition probability
                    'A_std': ndarray (K,K),
                }
            }
        }
    """
    with open(hmm_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = {'models': {}, 'best_n': None, 'method': None}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Method
        if line.startswith('Method,'):
            result['method'] = line.split(',', 1)[1].strip()
            i += 1
            continue

        # Suitable Model (BestN)
        if line.startswith('Suitable Model,'):
            result['best_n'] = int(line.split(',')[1].strip())
            i += 1
            continue

        # Model block
        m = re.match(r'^Model,(\d+)$', line)
        if m:
            k = int(m.group(1))
            model, i = _parse_model_block(lines, i + 1, k)
            result['models'][k] = model
            continue

        i += 1

    if result['best_n'] is None:
        raise ValueError(f"'Suitable Model' not found in {hmm_path}")

    return result


# hmm.csv spells the posterior parameters n / c / wpi; map to our names.
_POST_KEY = {'n': 'n', 'c': 'c', 'wpi': 'w_pi'}


def _parse_model_block(lines: list[str], start: int, k: int) -> tuple[dict, int]:
    """Parse a single Model block starting after 'Model,K' line."""
    model = {
        'lower_bound': None, 'lnZs': None,
        'kl': None, 'kl_pi': None, 'kl_diffusion': None, 'kl_b': None,
        'D_mean': None, 'D_std': None,
        'pi_mean': None, 'pi_std': None,
        'A_mean': None, 'A_std': None,
        # Posterior parameters, as written by AAS.  These are what make the
        # file self-checking: for K=1 the VBEM updates are closed form, so
        # n, c, w_pi and w_b determine the trajectory count, the step count
        # and sum(dx^2) that AAS actually analysed.  See
        # scripts/b1_recover_aas_inputs.py.
        'n': None, 'c': None, 'w_pi': None, 'w_b': None,
    }

    i = start
    while i < len(lines):
        line = lines[i].strip()

        # End of block: next Model or Suitable Model
        if re.match(r'^Model,\d+$', line) or line.startswith('Suitable Model,'):
            break

        # Value line: lower_bound, lnZs, kl, kl_pi, kl_diffusion, kl_b
        if line.startswith('Value,'):
            parts = line.split(',')
            if len(parts) >= 5:
                model['lower_bound'] = float(parts[1])
                model['lnZs'] = float(parts[2])
                model['kl'] = float(parts[3])
                model['kl_pi'] = float(parts[4]) if len(parts) > 4 else 0.0
                model['kl_diffusion'] = float(parts[5].strip()) if len(parts) > 5 else 0.0
                model['kl_b'] = float(parts[6]) if len(parts) > 6 else 0.0
            i += 1
            continue

        # Diffusion coefficient section
        if 'Diffusion coefficient' in line:
            # Skip header lines ("State,Average,Std,...")
            i += 2  # skip header + column names
            D_mean, D_std, pi_mean, pi_std = [], [], [], []
            for s in range(k):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split(',')
                if len(parts) >= 4:
                    D_mean.append(float(parts[1]))
                    D_std.append(float(parts[2]))
                    pi_mean.append(float(parts[3]))
                    pi_std.append(float(parts[4]) if len(parts) > 4 else 0.0)
                i += 1
            model['D_mean'] = np.array(D_mean)
            model['D_std'] = np.array(D_std)
            model['pi_mean'] = np.array(pi_mean)
            model['pi_std'] = np.array(pi_std)
            continue

        # Transition Probability section
        if line.startswith('Transition Probability,'):
            # Parse K rows of transition matrix
            A_mean = np.zeros((k, k))
            A_std = np.zeros((k, k))
            for s in range(k):
                i += 1
                if i >= len(lines):
                    break
                parts = lines[i].strip().split(',')
                # Format: state_num, A[s,0](Ave), A[s,0](Std), A[s,1](Ave), ...
                for j in range(k):
                    idx_mean = 1 + j * 2
                    idx_std = 2 + j * 2
                    if idx_mean < len(parts):
                        A_mean[s, j] = float(parts[idx_mean])
                    if idx_std < len(parts):
                        A_std[s, j] = float(parts[idx_std])
            model['A_mean'] = A_mean
            model['A_std'] = A_std
            i += 1
            continue

        # Posterior parameters: 'n,...', 'c,...', 'wpi,...', 'wb[i],...'
        m = re.match(r'^(n|c|wpi),(.+)$', line)
        if m and model[_POST_KEY[m.group(1)]] is None:
            model[_POST_KEY[m.group(1)]] = np.array(
                [float(x) for x in m.group(2).split(',')])
            i += 1
            continue

        m = re.match(r'^wb\[(\d+)\],(.+)$', line)
        if m:
            row = int(m.group(1))
            if model['w_b'] is None:
                model['w_b'] = np.full((k, k), np.nan)
            if row < k:
                vals = [float(x) for x in m.group(2).split(',')]
                if len(vals) != k:
                    raise ValueError(
                        f"hmm.csv Model {k}: wb[{row}] has {len(vals)} values, "
                        f"expected {k}")
                model['w_b'][row] = vals
            i += 1
            continue

        i += 1

    if model['w_b'] is not None and np.isnan(model['w_b']).any():
        missing = sorted(int(r) for r in np.unique(
            np.nonzero(np.isnan(model['w_b']))[0]))
        raise ValueError(
            f"hmm.csv Model {k}: transition prior rows {missing} are missing. "
            f"The block is incomplete; it is not filled in with a default.")

    return model, i


# ============================================================================
# AAS settings.csv reader
# ============================================================================

def load_aas_settings_csv(settings_path: str | Path) -> dict:
    """Parse AAS settings.csv (3-column: Name, Unit, Value).

    Returns dict with standardized keys and typed values.
    Raises ValueError for missing required keys.
    """
    settings_path = Path(settings_path)
    if not settings_path.exists():
        raise FileNotFoundError(f"settings.csv not found: {settings_path}")

    raw = {}
    with open(settings_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip first line (header: "Parameter name,Tracking2DSettings")
    # Second line: "Name,Unit,Value"
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 3:
            name = parts[0].strip()
            value = parts[2].strip()
            raw[name] = value
        elif len(parts) == 2:
            # Malformed line (e.g. "w_b_tilde,1" without unit)
            name = parts[0].strip()
            value = parts[1].strip()
            raw[name] = value

    # Convert to typed dict
    result = {}

    # Detection parameters
    result['detect_mode'] = raw.get('Detect mode', '')
    result['roi_size'] = int(raw['ROI'])
    result['scan_step'] = int(raw['Scan Length'])
    result['start_frame'] = int(raw['Analysis Start frame'])
    result['end_frame'] = int(raw['Analysis End frame'])
    result['intensity_threshold'] = float(raw['Intensity Threshold'])

    # Linkage parameters
    result['connect_mode'] = raw.get('Connect mode', '')
    result['connection_distance'] = float(raw['Connect distance'])
    result['connection_frames'] = int(raw['Connect Frame'])
    result['min_track_length'] = int(raw['Minimum trajectory length'])

    # Physical parameters
    result['pixel_size_um'] = float(raw['Distance per pixel[um]'])
    result['frame_interval_ms'] = float(raw['Time per frame[ms]'])
    result['frame_interval_s'] = result['frame_interval_ms'] / 1000.0

    # VBHMM parameters
    result['vbhmm_method'] = raw.get('HMM analysis method', 'SimpleVbSPT')
    result['vbhmm_min_frame'] = int(raw.get('HMM Minimum Frame', '1'))
    result['min_hidden'] = int(raw['Minimum number of states'])
    result['max_hidden'] = int(raw['Maximum number of states'])
    result['n_tilde'] = float(raw['n_tilde'])
    result['c_tilde'] = float(raw['c_tilde'])
    result['w_pi_tilde'] = float(raw['w_pi_tilde'])
    result['w_b_tilde'] = float(raw['w_b_tilde'])
    result['mag'] = float(raw['mag'])
    result['estimate_mode'] = raw.get('Estimate mode', 'MaxProb')

    # Boolean flags
    result['calc_kl_each'] = raw.get('Calculate KL for each trajectory', 'NO') == 'YES'
    result['add_per_traj'] = raw.get('Add hyper params to each trajectory', 'NO') == 'YES'

    return result
