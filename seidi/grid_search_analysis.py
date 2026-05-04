import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# =============== CONFIG ===============
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH      = os.path.join(SCRIPT_DIR, "results_grid", "grid_timecurves_riem_svmrbf_stieger2021.csv")
OUT_DIR       = os.path.join(SCRIPT_DIR, "plots_grid_summary")

FIG_NAME      = "grid_onefig_heatmap_and_timecurves"
CHANCE        = 0.25
YLIM          = (0.0, 1.0)

BAND_ORDER    = ["all", "theta", "mu", "mu_beta", "beta", "low_gamma"]
BAND_ALIASES  = {"theta":"theta", "low_gamma": "low_gamma", "mu-beta": "mu_beta", "mu beta": "mu_beta", "mu_beta": "mu_beta"}


def canon_band(x: str) -> str:
    x = str(x).strip()
    return BAND_ALIASES.get(x, x)


def annotate_cells(ax, M, fmt="{:.2f}", fs=7):
    nr, nc = M.shape
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            if np.isnan(v): continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color="white")


def plot_series(ax, x, y, label):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0: return
    if x.size == 1:
        ax.scatter(x, y, s=35, zorder=3, label=label)
    else:
        ax.plot(x, y, lw=1.5, marker="o", label=label)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df                   = pd.read_csv(CSV_PATH)
    required             = {"band","channel_set","win_sec","time_center_s","acc_mean"}
    missing              = required - set(df.columns)
    if missing: raise ValueError(f"CSV faltando colunas: {sorted(missing)}")

    df["band"]           = df["band"].map(canon_band)
    df["win_sec"]        = df["win_sec"].astype(float)
    df["time_center_s"]  = df["time_center_s"].astype(float)
    df["acc_mean"]       = df["acc_mean"].astype(float)

    chsets               = sorted(df["channel_set"].unique())
    wins                 = sorted(df["win_sec"].unique())
    bands_in_data        = sorted(df["band"].unique())
    bands                = [b for b in BAND_ORDER if b in bands_in_data]
    if not bands: raise ValueError(f"Nenhuma banda bateu com BAND_ORDER. Disponíveis: {bands_in_data}")

    hm                   = (df.groupby(["channel_set","band","win_sec"], observed=True)["acc_mean"].mean().reset_index())
    ln                   = (df.groupby(["channel_set","band","win_sec","time_center_s"], observed=True)["acc_mean"].mean().reset_index())

    nrows, ncols         = len(chsets), 1 + len(bands)
    fig, axes            = plt.subplots(nrows, ncols, figsize=(2.9*ncols, 2.5*nrows), constrained_layout=True)
    if nrows == 1: axes  = axes[np.newaxis, :]
    if ncols == 1: axes  = axes[:, np.newaxis]

    im_ref               = None

    for i, ch in enumerate(chsets):

        ax0              = axes[i, 0]
        piv              = (hm[hm["channel_set"] == ch]
                            .pivot_table(index="band", columns="win_sec", values="acc_mean", aggfunc="mean")
                            .reindex(index=bands, columns=wins))

        M                = piv.values.astype(float)
        im               = ax0.imshow(np.ma.masked_invalid(M), origin="lower", aspect="auto", vmin=YLIM[0], vmax=YLIM[1])
        im_ref           = im_ref or im

        ax0.set_title("band × win")
        ax0.set_xlabel("win_sec")
        ax0.set_ylabel(f"{ch}\nband")

        ax0.set_xticks(np.arange(len(wins)))
        ax0.set_xticklabels([f"{w:g}" for w in wins], fontsize=8)
        ax0.set_yticks(np.arange(len(bands)))
        ax0.set_yticklabels(bands, fontsize=9)

        annotate_cells(ax0, M, fmt="{:.2f}", fs=7)

        for j, band in enumerate(bands, start=1):
            ax           = axes[i, j]
            sub          = ln[(ln["channel_set"] == ch) & (ln["band"] == band)]
            if sub.empty:
                ax.set_axis_off()
                continue

            for win in wins:
                s          = sub[sub["win_sec"] == win].sort_values("time_center_s")
                if s.empty: 
                    continue
                plot_series(ax, s["time_center_s"].values, s["acc_mean"].values, label=f"{win:g}s")

            ax.axhline(CHANCE, color="k", ls="--", lw=1, alpha=0.8)
            ax.set_ylim(*YLIM)
            ax.grid(alpha=0.25)

            if i == 0: ax.set_title(band)
            if i == nrows - 1: ax.set_xlabel("time_center_s")
            if j == 1: ax.set_ylabel("acc")
            if i == 0 and j == ncols - 1:
                ax.legend(fontsize=7, frameon=False)

    cbar                 = fig.colorbar(im_ref, ax=axes[:, 0].ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("acc_mean")

    fig.suptitle("Grid summary: heatmap (band×win) + time curves (acc vs time)\nrows = channel_set", y=1.02)

    out_pdf              = os.path.join(OUT_DIR, FIG_NAME + ".pdf")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {out_pdf}")


if __name__ == "__main__":
    main()
