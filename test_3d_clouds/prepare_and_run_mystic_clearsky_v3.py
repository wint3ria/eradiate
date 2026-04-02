"""
prepare_and_run_mystic_clearsky_v3.py

IPRT C3 case 5 clear-sky reference using MYSTIC backward MC.

Uses mol_tau_file sca/abs to inject exact IPRT molecular OD profiles.
Uses mc_backward for radiance at nadir, TOA (z=30km), SZA=40.
Flat solar E0=1 -> mc.rad radiance = L/E0 (IPRT normalization).

mol_tau_file monochromatic format:
    col1 = altitude in km (layer bottom)
    col2 = optical depth of that layer
    top-down order

mc.rad with mc_polarisation:
    4 rows = I Q U V
    8 columns: ix iy vza vaz wl ? ? value
    Stokes value in last column (index 7)

umu convention: umu > 0 = looking downward (satellite)
For nadir sensor at TOA: umu = +1

Usage:
    python prepare_and_run_mystic_clearsky_v3.py \
        --uvspec /path/to/uvspec \
        --libdata /path/to/libradtran/data \
        --tau atmos_tau_cu.dat \
        [--photons 1000000] [--workdir ./run]
"""

import argparse
import os
import subprocess

import numpy as np
import pandas as pd

SZA_DEG = 40.0


def read_iprt_tau(tau_path):
    df = pd.read_csv(
        tau_path,
        comment="!",
        sep=r"\s+",
        header=None,
        names=[
            "z_bot",
            "z_top",
            "temp",
            "tau_abs067",
            "tau_abs213",
            "tau_abs110",
            "tau_ray067",
            "tau_aer067",
            "tau_aer213",
        ],
    )
    return df.sort_values("z_bot").reset_index(drop=True)


def write_mol_tau_file(df, col, path):
    """
    mol_tau_file monochromatic format (top-down):
        col1 = layer bottom altitude (km)
        col2 = layer optical depth
    """
    with open(path, "w") as f:
        for i in range(len(df) - 1, -1, -1):
            f.write(f"{df.z_bot.iloc[i]:.4f}  {df[col].iloc[i]:.8f}\n")
    print(f"Written {os.path.basename(path)}: total OD={df[col].sum():.5f}")
    return path


def write_solar(path):
    with open(path, "w") as f:
        f.write("669.0  1.0\n670.0  1.0\n671.0  1.0\n")
    print(f"Written solar file: {path}")
    return path


MYSTIC_INP = """\
# IPRT C3 clear-sky benchmark - molecular atmosphere only
# Case 5: SZA=40, nadir backward MC (umu=+1), zout=30km, albedo=0.2
# mol_tau_file injects exact IPRT Rayleigh and absorption ODs
# Flat solar E0=1 -> mc.rad = L/E0 (IPRT normalization)

rte_solver      montecarlo
mc_photons      {n_photons}
mc_polarisation
mc_backward

atmosphere_file {atmos_file}
wavelength      670.0 670.0
source solar    {solar_file}

# Inject exact IPRT molecular ODs (overrides internal Rayleigh/absorption)
mol_tau_file sca {sca_file}
mol_tau_file abs {abs_file}

# Rayleigh depolarization = 0 as per IPRT spec
rayleigh_depol  0.0

sza             {sza}
phi0            0.0
umu             1.0
phi             0.0
zout            30.0
albedo          0.2

quiet
"""


def write_inp(n_photons, atmos_file, sca_file, abs_file, solar_file, out_path):
    with open(out_path, "w") as f:
        f.write(
            MYSTIC_INP.format(
                n_photons=n_photons,
                atmos_file=os.path.abspath(atmos_file),
                sca_file=os.path.abspath(sca_file),
                abs_file=os.path.abspath(abs_file),
                solar_file=os.path.abspath(solar_file),
                sza=SZA_DEG,
            )
        )
    print(f"Written input file: {out_path}")


def run_uvspec(uvspec_bin, inp_path, libdata, workdir):
    env = {**os.environ, "LIBRADTRAN_DATA_FILES": libdata}
    stdout_path = os.path.join(workdir, "uvspec_stdout.dat")
    mc_rad_path = os.path.join(workdir, "mc.rad")
    print(f"Running: {uvspec_bin} < {inp_path}")
    with open(inp_path) as inp, open(stdout_path, "w") as out:
        r = subprocess.run(
            [uvspec_bin],
            stdin=inp,
            stdout=out,
            stderr=subprocess.PIPE,
            env=env,
            cwd=workdir,
        )
    stderr = r.stderr.decode()
    if stderr.strip():
        print("STDERR:", stderr)
    if r.returncode != 0:
        raise RuntimeError(f"uvspec failed: {r.returncode}")
    print("Done.")
    return mc_rad_path, stdout_path


def find_mc_rad(workdir):
    """Find the mc*.rad output file — may be mc.rad or mc{wl_idx}.rad."""
    import glob

    # Prefer mc.rad, fall back to mc{N}.rad
    candidates = glob.glob(os.path.join(workdir, "mc.rad")) + sorted(
        glob.glob(os.path.join(workdir, "mc[0-9]*.rad"))
    )
    if not candidates:
        raise FileNotFoundError(f"No mc*.rad file found in {workdir}")
    path = candidates[0]
    print(f"Reading: {path}")
    return path


def parse_mc_rad(mc_rad_path):
    """
    mc*.rad: 4 rows = I Q U V, Stokes value in last column (index 7).
    First two columns may be inf (no pixel index) — replace with 0.
    With flat solar E0=1: value = L/E0 = IPRT normalization.
    """
    rows = []
    with open(mc_rad_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Replace inf/-inf with 0 for pixel index columns
            vals = [
                0.0 if v.lower() in ("inf", "-inf", "nan") else float(v)
                for v in line.split()
            ]
            rows.append(vals)
    data = np.array(rows)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    labels = ["I", "Q", "U", "V"]
    stokes = {
        l: float(data[i, -1]) if i < len(data) else np.nan for i, l in enumerate(labels)
    }
    print("\n--- MYSTIC clear-sky result ---")
    print(f"  vza={data[0, 2]:.0f}  vaz={data[0, 3]:.0f}")
    for l in labels:
        print(f"  {l} = {stokes[l]:.6f}")
    I = stokes["I"]
    cos_sza = np.cos(np.deg2rad(SZA_DEG))
    I_erad = I / cos_sza
    print(f"\n  Raw MYSTIC I               : {I:.6f}")
    print(f"  Lambertian surface (MYSTIC) : {0.2 * cos_sza / np.pi:.6f}")
    print(
        f"  Atmospheric effect          : {(I - 0.2 * cos_sza / np.pi) / (0.2 * cos_sza / np.pi) * 100:+.1f}%"
    )
    print("\n  Converted to eradiate convention (/ cos(sza)):")
    print(f"  MYSTIC I_erad               : {I_erad:.6f}")
    print(f"  Lambertian surface (erad)   : {0.2 / np.pi:.6f}")
    print(f"  Eradiate clear-sky          : 0.064759  (ratio {0.064759 / I_erad:.4f})")
    print(f"  IPRT ref domain mean        : 0.062076  (ratio {0.062076 / I_erad:.4f})")
    return stokes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uvspec", required=True)
    p.add_argument("--libdata", required=True)
    p.add_argument("--tau", default="atmos_tau_cu.dat")
    p.add_argument("--photons", type=int, default=1_000_000)
    p.add_argument("--workdir", default="./mystic_run")
    p.add_argument(
        "--atmos_file", default="atmmod/afglus.dat", help="Relative to --libdata"
    )
    args = p.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    ld = os.path.abspath(args.libdata)
    wd = args.workdir

    atmos = os.path.join(ld, args.atmos_file)
    if not os.path.exists(atmos):
        raise FileNotFoundError(
            f"Not found: {atmos}\n"
            f"Available: {sorted(os.listdir(os.path.join(ld, 'atmmod')))}"
        )

    df = read_iprt_tau(args.tau)
    print(
        f"Read {len(df)} layers, tau_ray={df.tau_ray067.sum():.5f} "
        f"tau_abs={df.tau_abs067.sum():.5f}"
    )

    sca_file = write_mol_tau_file(df, "tau_ray067", os.path.join(wd, "mol_sca.dat"))
    abs_file = write_mol_tau_file(df, "tau_abs067", os.path.join(wd, "mol_abs.dat"))
    solar = write_solar(os.path.join(ld, "solar_unity.dat"))
    inp = os.path.join(wd, "mystic_clearsky.inp")

    write_inp(args.photons, atmos, sca_file, abs_file, solar, inp)
    mc_rad, _ = run_uvspec(args.uvspec, inp, ld, wd)
    mc_rad = find_mc_rad(wd)
    parse_mc_rad(mc_rad)
