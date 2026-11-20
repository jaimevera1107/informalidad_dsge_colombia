"""Write the LaTeX bodies of the posterior, fit and variance tables of the paper.

Usage:
    python scripts/tablas_paper.py --base resultados_m2/solo_trabajo_cuna_legal --out paper
The variance decomposition file must exist in the result folder (variance_decomposition.csv).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NAMES = {
    "theta": r"$\theta$ informal output share", "eta": r"$\eta$ informal labor elasticity",
    "rho_mu": r"$\rho_\mu$ trend growth", "rho_g": r"$\rho_g$ government", "rho_v": r"$\rho_\nu$ investment efficiency",
    "rho_psi": r"$\rho_\psi$ labor supply", "rho_om": r"$\rho_\omega$ formality wedge", "rho_x": r"$\rho_x$ absorption",
    "sig_mu": r"$\sigma_\mu$", "sig_g": r"$\sigma_g$", "sig_v": r"$\sigma_\nu$", "sig_psi": r"$\sigma_\psi$",
    "sig_om": r"$\sigma_\omega$", "sig_x": r"$\sigma_x$", "me_i": r"$\sigma_{e,I}$ investment (level)",
    "me_n": r"$\sigma_{e,N}$ employment (level)", "me_s": r"$\sigma_{e,s}$ logit informality (level)",
    "rho_zr": r"$\rho_r$ informal productivity", "sig_zr": r"$\sigma_r$", "me_w": r"$\sigma_{e,R}$ earnings ratio",
    "w_break": r"$b$ earnings shift from 2022Q1", "me_k": r"$\sigma_{e,K}$ capital--output ratio (level)",
}
FAMILY = {"beta": "Beta", "invgamma": r"Inv.\ gamma", "normal": "Normal"}
FIT = {"obs_y": "GDP growth", "obs_c": "Private consumption growth", "obs_i": "Investment growth",
       "obs_g": "Government consumption growth", "obs_n": "Employment growth",
       "obs_s": "Logit informality rate (level)", "obs_k": "Capital--output ratio (level)"}
VARIANCE = {"detrended_output": "Detrended output", "logit_informality": "Logit informality rate",
            "logit_informal_output": "Logit informal output share"}
SHOCK_COLUMNS = ("share_mu", "share_g", "share_v", "share_psi", "share_om", "share_x", "share_zr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    post = pd.read_csv(args.base / "posterior_summary.csv")
    (tables / "posterior.tex").write_text("\n".join(
        f"{NAMES[r.parameter]} & {FAMILY[r.prior]}({r.prior_mean:.3g}, {r.prior_sd:.3g}) & {r.post_median:.3f} & "
        f"[{r.p05:.3f}, {r.p95:.3f}] & {r.post_sd_over_prior_sd:.2f} & {r.rhat:.3f} & {r.ess_bulk:,.0f} \\\\"
        for r in post.itertuples()) + "\n")
    fit = pd.read_csv(args.base / "fit_one_step.csv")
    (tables / "fit.tex").write_text("\n".join(
        f"{FIT[r.observable]} & {r.sd_data:.4f} & {r.autocorr_data:.2f} & {r.rmse_model_over_sd:.3f} & "
        f"{r.rmse_ar1_over_sd:.3f} \\\\" for r in fit.itertuples()) + "\n")
    vd = pd.read_csv(args.base / "variance_decomposition.csv")
    (tables / "variance.tex").write_text("\n".join(
        f"{VARIANCE[r.variable]} & {r.horizon} & " + " & ".join(f"{100 * getattr(r, c):.1f}" for c in SHOCK_COLUMNS)
        + r" \\" for r in vd.itertuples()) + "\n")
    print(f"tables written to {tables}")


if __name__ == "__main__":
    main()
