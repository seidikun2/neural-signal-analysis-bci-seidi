"""
pseudo_online_run_riemann.py

Roda PseudoOnlineEvaluation (pseudo_online.py) no dataset BNCI2014_001,
com pipeline Riemann (bandpass -> cov -> tangentspace -> zscore -> SVM).

Treino/teste por sessão (sem partições):
- treino: 0train
- teste : 1test

Modelos:
- gate: IDLE vs TASK (binário, y_idle=1 se IDLE)
- task: classificador treinado apenas nas classes (1..4)

Salva:
- CSV com inferência no TEST (stage=idle_gate e stage=task)
- CSV com tempos de treino (models)
"""

import os, moabb
import pandas as pd

from moabb.datasets         import BNCI2014_001
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.svm            import SVC
from pyriemann.estimation   import Covariances
from pyriemann.tangentspace import TangentSpace

from pseudo_online          import PseudoOnlineEvaluation, ArrayFilter

moabb.set_log_level("info")

# ============================== CONFIG ============================== #
SUBJECTS        = [1]
WSIZE           = 2.0
WSTEP           = 0.2

LFREQ, HFREQ    = 1.0, 40.0
COV_ESTIMATOR   = "oas"

TRAIN_SESSIONS  = ("0train",)
TEST_SESSIONS   = ("0train",)

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
OUT_PREFIX      = os.path.join(SCRIPT_DIR, "bnci_riemann_two_stage")

# ============================ PIPELINES ============================ #
def infer_sfreq(dataset, subject):
    pre = dataset.get_data(subjects=[subject])
    for _, runs in pre.items():
        for _, dicts in runs.items():
            for _, raw in dicts.items():
                r = raw[0] if isinstance(raw, list) else raw
                return float(r.info["sfreq"])
    raise RuntimeError("Não consegui inferir sfreq do dataset.")


def build_riemann_pipelines(sfreq, cov_estimator="oas"):
    steps = [
        ("bp",  ArrayFilter(sfreq=sfreq, lfreq=LFREQ, hfreq=HFREQ)),
        ("cov", Covariances(estimator=cov_estimator)),
        ("ts",  TangentSpace(metric="riemann")),
        ("z",   StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True)),
    ]
    gate = {"RiemannTS+SVM_gate": Pipeline(steps)}
    task = {"RiemannTS+SVM_task": Pipeline(steps)}
    return gate, task


# =============================== MAIN =============================== #
def main():
    dataset = BNCI2014_001()

    sfreq = infer_sfreq(dataset, SUBJECTS[0])
    idle_gate_pipes, task_pipes = build_riemann_pipelines(sfreq, cov_estimator=COV_ESTIMATOR)

    ev = PseudoOnlineEvaluation(
        dataset                    = dataset,
        task_pipelines             = task_pipes,
        idle_gate_pipelines        = idle_gate_pipes,
        wsize                      = WSIZE,
        wstep                      = WSTEP,
        subjects                   = SUBJECTS,
        train_sessions             = TRAIN_SESSIONS,
        test_sessions              = TEST_SESSIONS,
        out_prefix                 = OUT_PREFIX,
        compute_task_for_all_windows=True,  # bom p/ threshold offline
        no_run                     = False,
    )
    ev.evaluate()

    # ------------------------ quick check outputs ------------------------ #
    for sub in SUBJECTS:
        f_csv = f"{OUT_PREFIX}-S{sub}.csv"
        f_mod = f"{OUT_PREFIX}-models-S{sub}.csv"

        print("\nArquivos gerados:")
        print(" -", f_csv, "OK" if os.path.exists(f_csv) else "(não achei)")
        print(" -", f_mod, "OK" if os.path.exists(f_mod) else "(não achei)")

        if os.path.exists(f_csv):
            df = pd.read_csv(f_csv, nrows=10)

            base_cols = [
                "stage", "pipeline", "session",
                "window_start", "window_end",
                "y_true", "y_hat", "max_proba",
                "p_idle", "p_task",
            ]
            base_cols = [c for c in base_cols if c in df.columns]

            proba_cols = [c for c in df.columns if c.startswith("p_") and c not in ("p_idle", "p_task")]

            print("\nExemplo (head):")
            print(df[base_cols + proba_cols[:8]].head(10))


if __name__ == "__main__":
    main()
