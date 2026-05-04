# pseudo_online.py
import time
import numpy as np
import pandas as pd
import mne

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import butter, lfilter

REST_LABEL = 0  # IDLE


# ======================================================================
# Windowing (label por protocolo via centro da janela)
# ======================================================================
class PseudoOnlineWindow:
    """
    Sliding windows + rótulo por protocolo (sem majority vote por amostra).

    Regra:
      - cada evento define um segmento ativo de MI:
            [ev_idx + interval[0], ev_idx + interval[1]]
      - cada janela recebe o rótulo pelo CENTRO da janela:
            center in segmento -> classe do evento
            else -> IDLE (0)
    """

    def __init__(self, raw, events, interval, task_ids, window_size, window_step, chan_list=None):
        self.raw         = raw
        self.events      = np.asarray(events)
        self.sfreq       = float(raw.info["sfreq"])
        self.task_ids    = task_ids
        self.chan_list   = chan_list

        self.window_size = int(round(window_size * self.sfreq))
        self.window_step = int(round(window_step * self.sfreq))
        self.t_start     = int(round(interval[0] * self.sfreq))
        self.t_end       = int(round(interval[1] * self.sfreq))

        self.segments    = self._build_segments()

    def _build_segments(self):
        valid_ids = set(self.task_ids.values())
        segs = []

        for ev_idx, _, ev_id in self.events:
            ev_id = int(ev_id)
            if ev_id not in valid_ids:
                continue

            seg_start = max(0, int(ev_idx) + self.t_start)
            seg_end   = min(self.raw.n_times, int(ev_idx) + self.t_end)
            if seg_end > seg_start:
                segs.append((seg_start, seg_end, ev_id))

        segs.sort(key=lambda x: x[0])
        return segs

    def _label_by_center(self, start_idx, end_idx):
        center = (start_idx + end_idx) // 2
        for seg_start, seg_end, lab in self.segments:
            if seg_start <= center < seg_end:
                return int(lab)
        return REST_LABEL

    def generate_windows(self):
        picks  = self.chan_list if self.chan_list is not None else None
        data   = self.raw.get_data(picks=picks)
        n_samp = data.shape[1]

        X, y, times = [], [], []
        for start_idx in range(0, n_samp - self.window_size, self.window_step):
            end_idx = start_idx + self.window_size
            X.append(data[:, start_idx:end_idx])
            y.append(self._label_by_center(start_idx, end_idx))
            times.append((start_idx / self.sfreq, end_idx / self.sfreq))

        return np.asarray(X), np.asarray(y), np.asarray(times)


# ======================================================================
# Simple bandpass transformer
# ======================================================================
class ArrayFilter(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq, lfreq, hfreq):
        self.sfreq = sfreq
        self.lfreq = lfreq
        self.hfreq = hfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        b, a = butter(N=4, Wn=[self.lfreq, self.hfreq], btype="bandpass", fs=self.sfreq)
        return np.asarray(lfilter(b, a, X, axis=-1))


# ======================================================================
# Pseudo-online evaluation: TREINO em 0train / TESTE em 1test (sem ratio)
# ======================================================================
class PseudoOnlineEvaluation:
    """
    Treino/teste por sessão (sem partições):
      - train_sessions = ("0train",)
      - test_sessions  = ("1test",)

    Dois modelos (pipeline 3 que você quer “pré-pronta”):
      - idle_gate_pipelines: binário (y_idle=1 se IDLE, 0 se TASK)
      - task_pipelines     : multi-classe treinado só em (1..4)

    Loga:
      - stage="idle_gate": p_<0/1> e (se quiser) p_idle/p_task
      - stage="task"     : p_<1..4> (e possivelmente p_0 se existir no pipe, mas normalmente não)
    """

    def __init__(
        self,
        dataset,
        task_pipelines,
        idle_gate_pipelines,
        wsize,
        wstep,
        subjects,
        train_sessions=("0train",),
        test_sessions=("1test",),
        out_prefix="results",
        compute_task_for_all_windows=True,
        no_run=False,
    ):
        self.dataset                  = dataset
        self.task_pipelines           = task_pipelines          # dict[str, Pipeline]
        self.idle_gate_pipelines      = idle_gate_pipelines     # dict[str, Pipeline]
        self.wsize                    = float(wsize)
        self.wstep                    = float(wstep)
        self.subjects                 = list(subjects)

        self.train_sessions           = tuple(train_sessions) if train_sessions is not None else None
        self.test_sessions            = tuple(test_sessions)  if test_sessions  is not None else None

        self.out_prefix               = out_prefix
        self.compute_task_for_all     = bool(compute_task_for_all_windows)
        self.no_run                   = bool(no_run)

        self.dataset_name             = getattr(dataset, "code", dataset.__class__.__name__)
        self.results_                 = []
        self.model_results_           = []

        if self.idle_gate_pipelines is None:
            raise AttributeError("idle_gate_pipelines não pode ser None (pipeline 3).")

    # ----------------------------- helpers ----------------------------- #
    @staticmethod
    def _raw_concat(raw_list):
        if not raw_list:
            raise ValueError("Raw list vazio.")
        if len(raw_list) == 1:
            r = raw_list[0]
            return mne.concatenate_raws(r) if isinstance(r, list) else r
        return mne.concatenate_raws(raw_list)

    @staticmethod
    def _mask_tasks(y):
        return y != REST_LABEL

    @staticmethod
    def _proba_cols_from_pipe(pipe, probs_1d):
        out = {}
        classes = getattr(pipe, "classes_", None)
        if classes is None or probs_1d is None:
            return out
        for cls, p in zip(classes, probs_1d):
            out[f"p_{int(cls)}"] = float(p)
        return out

    @staticmethod
    def _p_idle_task_from_bin(pipe, probs_1d):
        """
        Convenção do gate:
          y_idle = 1 se IDLE, 0 se TASK
        """
        classes = list(getattr(pipe, "classes_", []))
        if probs_1d is None or not classes:
            return None, None
        try:
            i_idle = classes.index(1)
            i_task = classes.index(0)
            return float(probs_1d[i_idle]), float(probs_1d[i_task])
        except Exception:
            return None, None

    def _append_row(self, *, subject, session, window, wstart, wend,
                    stage, pipeline_name, y_true, y_hat, max_proba, prob_cols, extra=None, t_predict=None):
        y_true = int(y_true)
        row = dict(
            dataset      = self.dataset_name,
            subject      = int(subject),
            session      = str(session),
            stage        = str(stage),
            pipeline     = str(pipeline_name),
            window       = int(window),
            window_start = float(wstart),
            window_end   = float(wend),
            y_true       = y_true,
            y_hat        = (None if y_hat is None else int(y_hat)),
            max_proba    = (None if max_proba is None else float(max_proba)),
            t_predict    = (None if t_predict is None else float(t_predict)),
        )
        row.update(prob_cols)
        if extra:
            row.update(extra)
        self.results_.append(row)

    def _log_train(self, *, stage, pipeline_name, session, t_train, n_train):
        self.model_results_.append(
            dict(
                dataset   = self.dataset_name,
                stage     = str(stage),
                pipeline  = str(pipeline_name),
                session   = str(session),
                t_train   = float(t_train),
                n_train   = int(n_train),
            )
        )

    # ----------------------------- data fetch ----------------------------- #
    def _collect_raws_by_session(self, pre_dict):
        """
        pre_dict = dataset.get_data(subjects=[subject]) (formato MOABB)
        Retorna: dict[session_name] -> list of raws (runs concatenáveis)
        """
        raws = {}
        for _, runs in pre_dict.items():
            for sess, dicts in runs.items():
                raws.setdefault(sess, [])
                for _, raw in dicts.items():
                    raws[sess].append(raw)
        return raws

    def _select_sessions(self, all_sessions, wanted):
        if wanted is None:
            return list(all_sessions)
        wanted = set(wanted)
        return [s for s in all_sessions if s in wanted]

    # ----------------------------- training ----------------------------- #
    def _fit_gate(self, X, y, session):
        y_idle = (y == REST_LABEL).astype(int)
        for name, pipe in self.idle_gate_pipelines.items():
            t0 = time.perf_counter()
            pipe.fit(X, y_idle)
            t1 = time.perf_counter()
            self._log_train(stage="idle_gate", pipeline_name=name, session=session, t_train=(t1 - t0), n_train=len(y_idle))

    def _fit_task(self, X, y, session):
        mask = self._mask_tasks(y)
        X_use, y_use = X[mask], y[mask]
        for name, pipe in self.task_pipelines.items():
            t0 = time.perf_counter()
            pipe.fit(X_use, y_use)
            t1 = time.perf_counter()
            self._log_train(stage="task", pipeline_name=name, session=session, t_train=(t1 - t0), n_train=len(y_use))

    # ----------------------------- inference ----------------------------- #
    def _infer_gate(self, subject, session, X, y, times):
        for w in range(len(X)):
            xw = np.asarray([X[w]])
            y_true = y[w]
            wstart, wend = times[w]

            for name, pipe in self.idle_gate_pipelines.items():
                t0 = time.perf_counter()
                probs = pipe.predict_proba(xw)[0]
                t1 = time.perf_counter()

                prob_cols = self._proba_cols_from_pipe(pipe, probs)
                p_idle, p_task = self._p_idle_task_from_bin(pipe, probs)

                y_hat = int(pipe.classes_[int(np.argmax(probs))])
                maxp  = float(np.max(probs))

                self._append_row(
                    subject=subject, session=session, window=w,
                    wstart=wstart, wend=wend,
                    stage="idle_gate", pipeline_name=name,
                    y_true=y_true, y_hat=y_hat, max_proba=maxp,
                    prob_cols=prob_cols,
                    extra=dict(p_idle=p_idle, p_task=p_task),
                    t_predict=(t1 - t0),
                )

    def _infer_task(self, subject, session, X, y, times):
        for w in range(len(X)):
            # se você quiser “simular custo online”, poderia pular janelas idle aqui
            if (not self.compute_task_for_all) and (y[w] == REST_LABEL):
                continue

            xw = np.asarray([X[w]])
            y_true = y[w]
            wstart, wend = times[w]

            for name, pipe in self.task_pipelines.items():
                t0 = time.perf_counter()
                probs = pipe.predict_proba(xw)[0]
                t1 = time.perf_counter()

                prob_cols = self._proba_cols_from_pipe(pipe, probs)
                y_hat = int(pipe.classes_[int(np.argmax(probs))])
                maxp  = float(np.max(probs))

                self._append_row(
                    subject=subject, session=session, window=w,
                    wstart=wstart, wend=wend,
                    stage="task", pipeline_name=name,
                    y_true=y_true, y_hat=y_hat, max_proba=maxp,
                    prob_cols=prob_cols,
                    extra=None,
                    t_predict=(t1 - t0),
                )

    # ----------------------------- main ----------------------------- #
    def evaluate(self):
        for subject in self.subjects:
            if subject not in self.dataset.subject_list:
                raise ValueError(f"Subject inválido: {subject}")

            pre = self.dataset.get_data(subjects=[subject])
            raws_by_sess = self._collect_raws_by_session(pre)
            all_sessions = list(raws_by_sess.keys())

            train_sess = self._select_sessions(all_sessions, self.train_sessions)
            test_sess  = self._select_sessions(all_sessions, self.test_sessions)

            if not train_sess:
                raise RuntimeError(f"Sem sessões de treino encontradas. Disponíveis={all_sessions} wanted={self.train_sessions}")
            if not test_sess:
                raise RuntimeError(f"Sem sessões de teste encontradas. Disponíveis={all_sessions} wanted={self.test_sessions}")

            # ---- TREINO: concatena todas as sessões de treino em um bloco ----
            Xtr_list, ytr_list = [], []
            for sess in train_sess:
                raw = self._raw_concat(raws_by_sess[sess])
                events, event_ids = mne.events_from_annotations(raw)
                wgen = PseudoOnlineWindow(raw, events, self.dataset.interval, event_ids, self.wsize, self.wstep)
                X, y, _ = wgen.generate_windows()
                Xtr_list.append(X); ytr_list.append(y)

            X_train = np.concatenate(Xtr_list, axis=0)
            y_train = np.concatenate(ytr_list, axis=0)

            if self.no_run:
                # útil p/ debug rápido
                return X_train, y_train

            # ---- fit gate + task (task treinado só em classes) ----
            self._fit_gate(X_train, y_train, session="+".join(train_sess))
            self._fit_task(X_train, y_train, session="+".join(train_sess))

            # ---- TESTE: roda inferência em cada sessão de teste ----
            for sess in test_sess:
                raw = self._raw_concat(raws_by_sess[sess])
                events, event_ids = mne.events_from_annotations(raw)
                wgen = PseudoOnlineWindow(raw, events, self.dataset.interval, event_ids, self.wsize, self.wstep)
                X_test, y_test, times_test = wgen.generate_windows()

                self._infer_gate(subject, sess, X_test, y_test, times_test)
                self._infer_task(subject, sess, X_test, y_test, times_test)

            # ---- save per subject ----
            if self.results_:
                pd.DataFrame(self.results_).to_csv(f"{self.out_prefix}-S{subject}.csv", index=False)
                self.results_.clear()

            if self.model_results_:
                pd.DataFrame(self.model_results_).to_csv(f"{self.out_prefix}-models-S{subject}.csv", index=False)
                self.model_results_.clear()

        return self
