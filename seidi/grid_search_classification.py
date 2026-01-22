import os, re, inspect, warnings, time, moabb, json
import numpy                  as np
import pandas                 as pd
from typing                   import List, Tuple
from moabb.paradigms          import MotorImagery
from moabb.datasets           import Stieger2021
from sklearn.base             import BaseEstimator, TransformerMixin
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import StratifiedKFold
from sklearn.preprocessing    import StandardScaler
from sklearn.svm              import SVC
from pyriemann.estimation     import Covariances
from pyriemann.tangentspace   import TangentSpace
warnings.filterwarnings("ignore")
moabb.set_log_level("info")

# =============== CONFIG ===============
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
OUT_DIR          = os.path.join(SCRIPT_DIR, "results_grid")
DATA_DIR         = r"/media/seidi/hd_seidi/dados_stieger"
SUBJECTS         = list(range(1, 25))
SESSIONS_USE     = [7]
INTERVAL         = [0.0, 2.75]
RESAMPLE_HZ      = 128
MAX_SPLITS       = 5
SEED             = 42

OUT_CSV          = os.path.join(OUT_DIR, "grid_timecurves_riem_svmrbf_stieger2021.csv")

WIN_LIST_SEC     = [0.5, 1.0, 1.5, 2.0]
STEP_LIST_SEC    = [0.2]

# >>> MUDANÇA 1: "all" SEM FILTRO (fmin/fmax=None)
BANDS            = [
    {"name": "all",      "fmin": None, "fmax": None},  # <-- sem filtro
    {"name": "delta",    "fmin": 0.5,  "fmax": 4.0},
    {"name": "theta",    "fmin": 4.0,  "fmax": 8.0},
    {"name": "mu",       "fmin": 8.0,  "fmax": 13.0},
    {"name": "mu_beta",  "fmin": 8.0,  "fmax": 30.0},
    {"name": "beta",     "fmin": 13.0, "fmax": 30.0},
    {"name": "lowgamma", "fmin": 30.0, "fmax": 45.0},
]

CHANNEL_SETS     = [
    {"name": "non_motor",   "chs": ["AF3", "AF4","Fp1", "Fpz", "Fp2", "PO7", "PO3", "POz", "PO4", "PO8","O1", "Oz", "O2"]},
    {"name": "motor_17",    "chs": ["FC3","FC1","FCz","FC2","FC4","C5","C3","C1","Cz","C2","C4","C6","CP3","CP1","CPz","CP2","CP4"]},
]

# ============== DATASET ===============
class Stieger2021Local(Stieger2021):

    def __init__(self, interval=[0, 3], sessions=None, fix_bads=True, data_dir=None):
        sig = inspect.signature(super().__init__)
        if "fix_bads" in sig.parameters:
            super().__init__(interval=interval, sessions=sessions, fix_bads=fix_bads)
        else:
            super().__init__(interval=interval, sessions=sessions)
        self.data_dir = data_dir

    def data_path(self, subject, **kwargs):
        if not self.data_dir or not os.path.isdir(self.data_dir):
            return []
        files = []
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".mat"):
                continue
            m = re.match(r"S(\d+)_Session_(\d+)\.mat", fname)
            if m and int(m.group(1)) == subject:
                ses = int(m.group(2))
                if self.sessions is None or ses in self.sessions:
                    files.append(os.path.join(self.data_dir, fname))
        return sorted(files)

# ============== PIPELINE ==============
class CAR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X)
        return X - X.mean(axis=1, keepdims=True)

def build_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("car", CAR()),
        ("cov", Covariances(estimator="oas")),
        ("ts",  TangentSpace(metric="riemann")),
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="rbf", gamma="scale", C=C, probability=True)),
    ])

# =============== HELPERS ===============
def list_available_sessions(ds: Stieger2021Local, subject: int) -> List[int]:
    out = set()
    for p in ds.data_path(subject):
        out.add(int(os.path.basename(p).split("_")[-1].split(".")[0]))
    return sorted(out)

def safe_stratified_cv(y: np.ndarray, max_splits: int, seed: int):
    _, cnt    = np.unique(y, return_counts=True)
    min_cnt   = int(cnt.min())
    if min_cnt < 2: return None
    n_splits  = min(max_splits, min_cnt)
    if n_splits < 2: return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def make_windows(n_times: int, sfreq: float, t0: float, win_sec: float, step_sec: float):
    win  = int(round(win_sec  * sfreq))
    step = int(round(step_sec * sfreq))
    if win < 2: return []
    if step < 1: step = 1
    out = []
    for a in range(0, n_times - win + 1, step):
        b      = a + win
        center = (a + b - 1) / 2.0
        out.append((a, b, float(t0 + center / sfreq)))
    return out

# >>> MUDANÇA 2: retornar também a DISTRIBUIÇÃO (scores por fold) por janela
def timecurve_acc_over_folds(
    X: np.ndarray, y: np.ndarray,
    win_sec: float, step_sec: float,
    interval_t0: float, sfreq: float,
    cv
) -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
    n_trials, _, n_times = X.shape
    windows              = make_windows(n_times, sfreq, interval_t0, win_sec, step_sec)
    if not windows:
        return np.array([]), np.array([]), []

    t_centers            = np.array([w[2] for w in windows], float)
    acc_mean             = np.zeros(len(windows), float)
    acc_folds_all         = []  # list (len=windows) de listas (len=n_folds)

    for w_id, (a, b, _) in enumerate(windows):
        fold_scores = []
        for tr_idx, te_idx in cv.split(X, y):
            pipe = build_pipeline(C=1.0)
            pipe.fit(X[tr_idx, :, a:b], y[tr_idx])
            yhat = pipe.predict(X[te_idx, :, a:b])
            fold_scores.append(float(np.mean(yhat == y[te_idx])))

        acc_folds_all.append(fold_scores)
        acc_mean[w_id] = float(np.mean(fold_scores))

    return t_centers, acc_mean, acc_folds_all

# ================= MAIN =================
if __name__ == "__main__":

    os.makedirs(OUT_DIR, exist_ok=True)

    rows   = []
    run_id = 0

    for subj in SUBJECTS:
        ds_list              = Stieger2021Local(interval=INTERVAL, sessions=SESSIONS_USE, data_dir=DATA_DIR)
        ds_list.subject_list = [subj]
        avail                = list_available_sessions(ds_list, subj)
        use_sessions         = sorted(set(avail) & set(SESSIONS_USE))
        print(f"\n=== Subject {subj} | Sessions: {use_sessions} ===")

        for ses in use_sessions:
            ds                = Stieger2021Local(interval=INTERVAL, sessions=[ses], data_dir=DATA_DIR)
            ds.subject_list    = [subj]
            print(f"\n--- Subject {subj} | Session {ses} ---")

            for band in BANDS:
                fmin = band["fmin"]
                fmax = band["fmax"]

                for chset in CHANNEL_SETS:
                    chs = chset["chs"]

                    # >>> MUDANÇA 1b: só passa fmin/fmax se existirem
                    mi_kwargs = dict(n_classes=4, resample=RESAMPLE_HZ, channels=chs)
                    if fmin is not None: mi_kwargs["fmin"] = float(fmin)
                    if fmax is not None: mi_kwargs["fmax"] = float(fmax)
                    paradigm = MotorImagery(**mi_kwargs)

                    try:
                        X, y, _ = paradigm.get_data(ds, subjects=[subj])
                        if X is None or X.shape[0] == 0:
                            print(f"[WARN] vazio band={band['name']} ch={chset['name']}")
                            continue

                        cv = safe_stratified_cv(y, MAX_SPLITS, SEED)
                        if cv is None:
                            print(f"[WARN] sem CV band={band['name']} ch={chset['name']}")
                            continue

                        for win_sec in WIN_LIST_SEC:
                            for step_sec in STEP_LIST_SEC:
                                run_id  += 1
                                t_start = time.perf_counter()

                                t_cent, acc_mean, acc_folds_all = timecurve_acc_over_folds(
                                    X=X, y=y,
                                    win_sec=float(win_sec),
                                    step_sec=float(step_sec),
                                    interval_t0=float(INTERVAL[0]),
                                    sfreq=float(RESAMPLE_HZ),
                                    cv=cv,
                                )

                                if t_cent.size == 0:
                                    continue

                                elapsed = time.perf_counter() - t_start

                                for w_i in range(len(t_cent)):
                                    fold_scores = acc_folds_all[w_i]

                                    rows.append({
                                        "run_id"        : int(run_id),
                                        "subject"       : int(subj),
                                        "session"       : int(ses),
                                        "band"          : band["name"],
                                        "fmin"          : (np.nan if fmin is None else float(fmin)),
                                        "fmax"          : (np.nan if fmax is None else float(fmax)),
                                        "channel_set"   : chset["name"],
                                        "n_channels"    : int(len(chs)),
                                        "win_sec"       : float(win_sec),
                                        "step_sec"      : float(step_sec),
                                        "time_center_s" : float(t_cent[w_i]),

                                        # média e distribuição
                                        "acc_mean"      : float(acc_mean[w_i]),
                                        "acc_folds_json": json.dumps(fold_scores),

                                        "n_trials"      : int(X.shape[0]),
                                        "n_splits"      : int(cv.get_n_splits()),
                                        "elapsed_s"     : float(elapsed),
                                        "pipeline"      : "riem+svm_rbf_time_resolved",
                                    })

                                print(f"[OK] band={band['name']} ch={chset['name']} "
                                      f"win={win_sec:.2f}s step={step_sec:.2f}s -> mean(acc)={acc_mean.mean():.3f}")

                    except Exception as e:
                        print(f"[ERROR] subj={subj} ses={ses} band={band['name']} ch={chset['name']} -> {e}")

    if not rows:
        raise RuntimeError("Nada foi gerado. Ajuste grids / DATA_DIR / canais / bandas.")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[FINAL] Salvo em: {OUT_CSV}")
    print(df.head())
