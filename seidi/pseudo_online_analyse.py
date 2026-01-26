"""
pseudo_online_analyse.py

1 figura por pipeline/stage (primeiro 1 minuto):
- topo: y_true (step plot) [no espaço correto do stage]
- meio: probabilidades p_k(t) (suavizadas)
- desenha linha horizontal do threshold (thr)
- base: strips coloridos para y_true e y_hat(thresholded)
- direita: matriz de confusão (no mesmo 1 minuto)

Compatível com CSVs da PseudoOnlineEvaluation revisada:
- stage="idle_gate": problema binário (0=TASK, 1=IDLE)
    - y_true original (0/1/2/3/4) -> gate: 1 se IDLE (y_true==0), senão 0
    - probs: prefere (p_task,p_idle). fallback para (p_0,p_1)
- stage="task": problema multi-classe (1..4)
    - avalia apenas janelas onde y_true != 0 (IDLE não entra na métrica)
    - probs: p_<k> numérico (tipicamente 1..4). (p_0 pode existir -> ignorado)

Extra:
- y_hat é SEMPRE recalculado a partir das probabilidades + limiar:
      se max_proba < thr => UNCERTAIN_LABEL
- Acc e CM refletem esse y_hat.
- ROC por varredura de threshold:
    * idle_gate: ROC binária (IDLE positivo)
    * task: curva "coverage vs accuracy" (porque ROC multi-classe com abstain é ambígua)
          - coverage(thr) = fração de janelas (de task) não-incertas
          - acc(thr)      = acurácia nas janelas não-incertas
"""

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================== CONFIG ============================== #
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = SCRIPT_DIR
OUT_DIR      = os.path.join(SCRIPT_DIR, "analysis_out")
os.makedirs(OUT_DIR, exist_ok=True)

PREFIXES       = ["bnci_riemann_two_stage", "riemann_two_stage"]
MINUTES        = 1.0
SMOOTH_SAMPLES = 7

DEFAULT_THR    = 0.4
THR_BY_STAGE   = {"idle_gate": 0.40, "task": 0.40}

UNCERTAIN_LABEL = -2   # classe extra: baixa confiança
IDLE_LABEL      = 0    # no y_true original
TASK_GATE_LABEL = 0    # no gate space
IDLE_GATE_LABEL = 1    # no gate space

# thresholds para curva ROC / coverage-acc
THR_GRID = np.linspace(0.0, 1.0, 101)

# ============================== IO ============================== #
def find_csvs(prefixes=PREFIXES, base_dir=RESULTS_DIR):
    out = []
    for pref in prefixes:
        out += glob.glob(os.path.join(base_dir, f"{pref}-S*.csv"))
    return sorted(set(out))

def parse_prefix(path):
    return os.path.basename(path).rsplit("-S", 1)[0]

def parse_subject(path):
    b = os.path.basename(path)
    try:
        return int(b.split("-S", 1)[1].split(".csv", 1)[0])
    except Exception:
        return None

def load_csv(path):
    df = pd.read_csv(path)
    df["prefix"]  = parse_prefix(path)
    df["subject"] = parse_subject(path)

    for c in ["window_start","window_end","y_true","y_hat","max_proba","t_predict"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["mode","stage","pipeline","session"]:
        df[c] = df[c].astype(str) if c in df.columns else "NA"

    for c in df.columns:
        if c.startswith("p_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ============================== helpers ============================== #
def pcols_numeric(df):
    """Somente p_<int> (exclui p_idle/p_task)."""
    cols = []
    for c in df.columns:
        if not c.startswith("p_"):
            continue
        suf = c.split("_", 1)[1]
        if re.fullmatch(r"\d+", str(suf)):
            cols.append(c)
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))

def k_of(col):
    try:
        suf = col.split("_", 1)[1]
        return int(suf) if re.fullmatch(r"\d+", str(suf)) else None
    except Exception:
        return None

def slice_first_minute(df, minutes=MINUTES):
    df = df.dropna(subset=["window_start","window_end","y_true"]).copy()
    if len(df) == 0:
        return df
    t0 = float(df["window_start"].min())
    t1 = t0 + 60.0 * float(minutes)
    return df[(df["window_start"] >= t0) & (df["window_start"] <= t1)].copy()

def leading_smooth(df, win=SMOOTH_SAMPLES):
    df = df.sort_values("window_start").copy()
    cols = pcols_numeric(df) + [c for c in ["p_idle","p_task"] if c in df.columns]
    for c in cols:
        df[c] = df[c].rolling(win, center=True, min_periods=1).mean()
    return df

def step_seconds(df):
    t = np.sort(df["window_start"].to_numpy(float))
    if len(t) < 2:
        return float(df["window_end"].iloc[0] - df["window_start"].iloc[0])
    dt = np.diff(t)
    dt = dt[dt > 0]
    return float(np.median(dt)) if len(dt) else float(df["window_end"].iloc[0] - df["window_start"].iloc[0])

def colors_for(labels):
    labels = sorted(set(int(x) for x in labels))
    cmap = plt.cm.tab10 if len(labels) <= 10 else plt.cm.tab20
    out = {}
    for k in labels:
        if k == UNCERTAIN_LABEL:
            out[k] = "0.90"
        elif k == IDLE_GATE_LABEL:
            out[k] = "0.25"
        elif k == TASK_GATE_LABEL:
            out[k] = "0.65"
        else:
            out[k] = cmap(int(k) % cmap.N)
    return out

def draw_strip(ax, t, step, labs, y_level, cmap):
    for t0, lab in zip(t, labs):
        if not np.isfinite(lab):
            col = "0.85"
        else:
            col = cmap.get(int(lab), "0.6")
        ax.plot([t0, t0 + step], [y_level, y_level], lw=10, color=col, solid_capstyle="butt")

def cm_simple(y_true, y_hat, labels):
    y_true = np.asarray(y_true, float)
    y_hat  = np.asarray(y_hat,  float)
    m = np.isfinite(y_true) & np.isfinite(y_hat)
    y_true = y_true[m].astype(int)
    y_hat  = y_hat[m].astype(int)

    labels = list(labels)
    idx = {k:i for i,k in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for yt, yh in zip(y_true, y_hat):
        if yt in idx and yh in idx:
            cm[idx[yt], idx[yh]] += 1
    return cm

# ============================== stage logic ============================== #
def stage_name(df):
    return str(df["stage"].iloc[0]) if "stage" in df.columns and len(df) else "NA"

def ytrue_stage(df):
    """
    Retorna:
      y_plot: y_true no espaço do stage
      eval_mask: onde vale calcular métrica/CM
    """
    st = stage_name(df)
    y = df["y_true"].to_numpy(float)

    if st == "idle_gate":
        y_gate = (y == IDLE_LABEL).astype(float)  # 1=IDLE, 0=TASK
        return y_gate, np.isfinite(y_gate)

    if st == "task":
        # avalia só quando y_true != 0
        m = np.isfinite(y) & (y != IDLE_LABEL)
        return y.copy(), m

    return y.copy(), np.isfinite(y)

def prob_matrix_and_labels(df):
    """
    Retorna:
      P: (n, C) probs
      ks: (C,) labels correspondentes às colunas
      stage
    """
    st = stage_name(df)

    if st == "idle_gate":
        if ("p_task" in df.columns) and ("p_idle" in df.columns):
            P = df[["p_task", "p_idle"]].to_numpy(float)
            ks = np.array([TASK_GATE_LABEL, IDLE_GATE_LABEL], int)
            return P, ks, st

        cols = [c for c in ["p_0","p_1"] if c in df.columns]
        if len(cols) == 2:
            P = df[cols].to_numpy(float)
            ks = np.array([k_of(c) for c in cols], int)
            return P, ks, st

        return None, None, st

    # task stage: usa p_<int>
    cols = pcols_numeric(df)
    if not cols:
        return None, None, st

    ks = np.array([k_of(c) if k_of(c) is not None else -999 for c in cols], int)
    P = df[cols].to_numpy(float)
    return P, ks, st

def safe_argmax_and_max(P, ks):
    """
    Robusto a linhas all-NaN:
      - se linha all-NaN => y_argmax=NaN, mx=NaN
    """
    P = np.asarray(P, float)
    ks = np.asarray(ks, int)

    all_nan = np.all(~np.isfinite(P), axis=1)
    P2 = np.where(np.isfinite(P), P, -np.inf)

    arg = np.argmax(P2, axis=1)
    mx  = np.max(P2, axis=1)

    y = ks[arg].astype(float)
    y[all_nan]  = np.nan
    mx[all_nan] = np.nan

    # se mx ficou -inf ou NaN, zera
    bad = ~np.isfinite(mx)
    y[bad]  = np.nan
    mx[bad] = np.nan
    return y, mx

def yhat_thresholded(df, thr):
    """
    Sempre calcula y_hat a partir das probs + limiar.
    Retorna (yhat, mx).
    """
    P, ks, st = prob_matrix_and_labels(df)
    if P is None:
        return np.full(len(df), np.nan), np.full(len(df), np.nan)

    y_argmax, mx = safe_argmax_and_max(P, ks)
    yhat = y_argmax.copy()

    m = np.isfinite(mx)
    yhat[m & (mx < thr)] = float(UNCERTAIN_LABEL)
    return yhat, mx

def classes_for_stage(df):
    st = stage_name(df)
    if st == "idle_gate":
        return [TASK_GATE_LABEL, IDLE_GATE_LABEL, UNCERTAIN_LABEL]

    if st == "task":
        cols = pcols_numeric(df)
        ks = [k_of(c) for c in cols]
        ks = [k for k in ks if k is not None and k >= 1]
        return sorted(set(ks + [UNCERTAIN_LABEL]))

    cols = pcols_numeric(df)
    ks = [k_of(c) for c in cols if k_of(c) is not None]
    return sorted(set(ks + [UNCERTAIN_LABEL]))

# ============================== ROC / Curves ============================== #
def roc_idle_gate(df):
    """
    ROC binária para stage idle_gate:
      positivo = IDLE (label 1)
      score = p_idle (se existir) senão prob da classe 1
    Retorna fpr, tpr (arrays) e auc (float).
    """
    st = stage_name(df)
    if st != "idle_gate":
        return None

    y_true_gate, eval_mask = ytrue_stage(df)  # 1=IDLE
    y_true = y_true_gate[eval_mask].astype(int)

    if "p_idle" in df.columns:
        score = df.loc[eval_mask, "p_idle"].to_numpy(float)
    else:
        # tenta p_1
        if "p_1" not in df.columns:
            return None
        score = df.loc[eval_mask, "p_1"].to_numpy(float)

    # remove NaNs
    m = np.isfinite(score)
    y_true = y_true[m]
    score  = score[m]

    if len(np.unique(y_true)) < 2:
        return None

    # calcula ROC manual (sem sklearn)
    order = np.argsort(-score)
    y = y_true[order]
    s = score[order]

    P = (y == 1).sum()
    N = (y == 0).sum()
    if P == 0 or N == 0:
        return None

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    last = None
    for yi, si in zip(y, s):
        if last is None or si != last:
            tpr.append(tp / P)
            fpr.append(fp / N)
            last = si
        if yi == 1:
            tp += 1
        else:
            fp += 1

    tpr.append(1.0)
    fpr.append(1.0)

    fpr = np.asarray(fpr, float)
    tpr = np.asarray(tpr, float)
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc

def coverage_acc_task(df, thr_grid=THR_GRID):
    """
    Para stage=task:
      coverage(thr) = fração de janelas (de task) com max_proba >= thr
      acc(thr)      = acurácia nessas janelas
    """
    st = stage_name(df)
    if st != "task":
        return None

    ytrue_plot, eval_mask = ytrue_stage(df)
    ytrue = ytrue_plot[eval_mask].astype(int)

    P, ks, _ = prob_matrix_and_labels(df)
    if P is None:
        return None

    # argmax e max_proba (sem threshold)
    y_argmax, mx = safe_argmax_and_max(P, ks)
    y_argmax = y_argmax[eval_mask]
    mx       = mx[eval_mask]

    # remove NaNs
    m = np.isfinite(y_argmax) & np.isfinite(mx)
    ytrue = ytrue[m]
    y_argmax = y_argmax[m].astype(int)
    mx = mx[m]

    if len(ytrue) == 0:
        return None

    cov = []
    acc = []
    for thr in thr_grid:
        keep = mx >= thr
        cov.append(float(keep.mean()) if len(keep) else np.nan)
        if keep.sum() == 0:
            acc.append(np.nan)
        else:
            acc.append(float((ytrue[keep] == y_argmax[keep]).mean()))
    return np.asarray(cov, float), np.asarray(acc, float)

# ============================== plotting ============================== #
def plot_probs(ax, df, t):
    st = stage_name(df)
    cmap = None

    if st == "idle_gate":
        # cores simples (sem exigir cmap)
        if "p_task" in df.columns:
            ax.plot(t, df["p_task"].to_numpy(float), lw=1.8, label="p_task")
        if "p_idle" in df.columns:
            ax.plot(t, df["p_idle"].to_numpy(float), lw=1.8, label="p_idle")
        for c in ["p_0", "p_1"]:
            if c in df.columns:
                ax.plot(t, df[c].to_numpy(float), lw=1.2, label=c)
        return

    # task
    cols = pcols_numeric(df)
    cls = [k_of(c) for c in cols]
    cls = [k for k in cls if k is not None]
    cmap = colors_for(cls + [UNCERTAIN_LABEL])

    for c in cols:
        k = k_of(c)
        if k is None:
            continue
        if st == "task" and k == 0:
            continue
        ax.plot(t, df[c].to_numpy(float), lw=1.8, color=cmap.get(int(k), "0.6"), label=f"p_{k}")

def plot_group(dfp):
    # recorta e suaviza
    gg = slice_first_minute(dfp)
    gg = leading_smooth(gg)
    gg = gg.dropna(subset=["window_start","window_end","y_true"]).sort_values("window_start").copy()
    if len(gg) == 0:
        return

    st = stage_name(gg)
    thr = float(THR_BY_STAGE.get(st, DEFAULT_THR))

    t = gg["window_start"].to_numpy(float)
    step = step_seconds(gg)

    ytrue_plot, eval_mask = ytrue_stage(gg)
    yhat_plot, mx = yhat_thresholded(gg, thr)  # <-- y_hat já reflete o limiar

    classes = classes_for_stage(gg)
    cmap = colors_for(classes)

    # métricas com y_hat thresholded
    ytrue_eval = np.where(eval_mask, ytrue_plot, np.nan)
    yhat_eval  = np.where(eval_mask, yhat_plot,  np.nan)
    m = np.isfinite(ytrue_eval) & np.isfinite(yhat_eval)
    acc = float((ytrue_eval[m].astype(int) == yhat_eval[m].astype(int)).mean()) if m.sum() else np.nan
    cm = cm_simple(ytrue_eval, yhat_eval, labels=classes)

    # ----- layout principal -----
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 1.3], height_ratios=[1.0, 3.0], wspace=0.15, hspace=0.05)

    ax_true = fig.add_subplot(gs[0, 0])
    ax_prob = fig.add_subplot(gs[1, 0], sharex=ax_true)
    ax_cm   = fig.add_subplot(gs[:, 1])

    # y_true
    ax_true.step(t, ytrue_plot, where="post", lw=1.6, color="k", label="y_true")
    ax_true.set_ylabel("y_true")
    ax_true.grid(True, alpha=0.2)
    ax_true.legend(loc="upper right", fontsize=9)
    plt.setp(ax_true.get_xticklabels(), visible=False)

    # probs
    plot_probs(ax_prob, gg, t)

    # threshold line
    ax_prob.axhline(thr, lw=1.2, linestyle="--", color="k", alpha=0.7, label=f"thr={thr:.2f}")

    ax_prob.set_ylim(-0.28, 1.02)
    ax_prob.set_ylabel("p_k")
    ax_prob.set_xlabel("window_start (s)")
    ax_prob.grid(True, alpha=0.25)
    ax_prob.legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.9)

    # strips
    draw_strip(ax_prob, t, step, ytrue_plot, -0.10, cmap)
    draw_strip(ax_prob, t, step, yhat_plot,  -0.20, cmap)
    ax_prob.text(0.01, 0.06, "strip y_true", transform=ax_prob.transAxes, fontsize=9, va="bottom")
    ax_prob.text(0.01, 0.01, "strip y_hat (thr)", transform=ax_prob.transAxes, fontsize=9, va="bottom")

    # título simples: pipeline + acc + subject
    meta = gg.iloc[0]
    subj = int(meta.get("subject", -1)) if pd.notna(meta.get("subject", np.nan)) else -1
    pip  = str(meta.get("pipeline", "NA"))
    ax_true.set_title(f"{pip} | S{subj} | acc={acc:.3f} | stage={st}", fontsize=12)

    # CM
    ax_cm.imshow(cm, aspect="auto")
    ax_cm.set_title("Confusion", fontsize=11)
    ax_cm.set_xlabel("y_hat")
    ax_cm.set_ylabel("y_true")
    ax_cm.set_xticks(range(len(classes)))
    ax_cm.set_yticks(range(len(classes)))
    ax_cm.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax_cm.set_yticklabels(classes, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    # salva
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", f"{pip}_S{subj}_{st}_{meta.get('session','NA')}")
    out_path = os.path.join(OUT_DIR, f"plot_{safe}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.show()
    plt.close(fig)
    print("Salvo:", out_path)

    # ----- curva ROC / coverage-acc (fig separada) -----
    fig2 = plt.figure(figsize=(6.5, 5.2))
    ax2 = fig2.add_subplot(111)

    if st == "idle_gate":
        roc = roc_idle_gate(gg)
        if roc is not None:
            fpr, tpr, auc = roc
            ax2.plot(fpr, tpr, lw=2.0)
            ax2.plot([0, 1], [0, 1], lw=1.0, linestyle="--", alpha=0.6)
            ax2.set_xlabel("FPR")
            ax2.set_ylabel("TPR")
            ax2.set_title(f"ROC (IDLE=pos) | {pip} | S{subj} | AUC={auc:.3f}")
            ax2.grid(True, alpha=0.25)
        else:
            ax2.text(0.5, 0.5, "ROC indisponível\n(classe única ou probs faltando)", ha="center", va="center")
            ax2.set_axis_off()

    elif st == "task":
        ca = coverage_acc_task(gg, THR_GRID)
        if ca is not None:
            cov, accs = ca
            ax2.plot(cov, accs, lw=2.0)
            ax2.set_xlabel("Coverage (mx >= thr)")
            ax2.set_ylabel("Accuracy (onde não-incerto)")
            ax2.set_title(f"Coverage–Accuracy | {pip} | S{subj}")
            ax2.grid(True, alpha=0.25)
        else:
            ax2.text(0.5, 0.5, "Curva indisponível\n(probs faltando)", ha="center", va="center")
            ax2.set_axis_off()
    else:
        ax2.text(0.5, 0.5, "Curva não definida\npara este stage", ha="center", va="center")
        ax2.set_axis_off()

    safe2 = re.sub(r"[^A-Za-z0-9_\-]+", "_", f"{pip}_S{subj}_{st}_curve_{meta.get('session','NA')}")
    out_path2 = os.path.join(OUT_DIR, f"curve_{safe2}.png")
    plt.tight_layout()
    plt.savefig(out_path2, dpi=170)
    plt.show()
    plt.close(fig2)
    print("Salvo:", out_path2)


# ============================== grouping ============================== #
def iter_groups(df):
    cols = ["prefix","subject","stage","pipeline","session"]
    for _, g in df.groupby(cols):
        yield g.copy()

# ============================== main ============================== #
def main():
    csvs = find_csvs()
    if not csvs:
        print("Não achei CSVs. Ajuste PREFIXES/RESULTS_DIR.")
        return

    df = pd.concat([load_csv(p) for p in csvs], ignore_index=True)
    df = df.dropna(subset=["subject"]).copy()
    if len(df) == 0:
        print("Sem subject válido.")
        return

    # fixa no primeiro sujeito para não misturar (como você vinha fazendo)
    sub0 = int(df["subject"].iloc[0])
    df = df[df["subject"] == sub0].copy()

    groups = list(iter_groups(df))
    if not groups:
        print("Não achei grupos por prefix/subject/stage/pipeline/session.")
        return

    for g in groups:
        plot_group(g)

if __name__ == "__main__":
    main()
