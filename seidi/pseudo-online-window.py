import time, mne
import numpy        as np
import pandas       as pd
from sklearn.base   import BaseEstimator, TransformerMixin
from scipy.signal   import butter, lfilter

REST_LABEL          = 0
REJECT_LABEL        = -1

class PseudoOnlineWindow:
    """
    Creates pseudo-online windows and their labels (majority label inside each window).

    labels per-sample:
      - default = REST_LABEL (idle)
      - for each event: interval [event + interval[0], event + interval[1]] gets event_id
    """

    def __init__(self, raw, events, interval, task_ids, window_size, window_step, chan_list=None):
        self.raw            = raw
        self.events         = events
        self.interval       = interval
        self.sfreq          = float(raw.info["sfreq"])
        self.task_ids       = task_ids
        self.chan_list      = chan_list

        self.window_size    = int(window_size * self.sfreq)
        self.window_step    = int(window_step * self.sfreq)
        self.t_start        = int(interval[0] * self.sfreq)
        self.t_end          = int(interval[1] * self.sfreq)

        self.labels         = self._generate_labels()

    def _generate_labels(self):
        n_samples           = self.raw.n_times
        labels              = np.zeros(n_samples, dtype=int)

        valid_ids           = set(self.task_ids.values())
        for ev_idx, _, ev_id in self.events:
            if ev_id not in valid_ids:
                continue

            start           = max(0, ev_idx + self.t_start)
            stop            = min(n_samples, ev_idx + self.t_end)
            labels[start:stop] = ev_id

        return labels

    @staticmethod
    def _majority_label(window_labels):
        count                 = np.bincount(window_labels)
        major                 = int(np.argmax(count))
        prop_major            = count[major] / len(window_labels)
 
        n_classes             = len(np.unique(window_labels))
        draw_prop             = 1 / n_classes

        return major if prop_major != draw_prop else int(window_labels[-1])

    def generate_windows(self):
        X, y, times           = [], [], []
 
        picks                 = self.chan_list if self.chan_list is not None else None
        data                  = self.raw.get_data(picks=picks)  # shape: (n_ch, n_samples)
        n_samples             = data.shape[1]

        for start_idx in range(0, n_samples - self.window_size, self.window_step):
            end_idx           = start_idx + self.window_size

            window_data       = data[:, start_idx:end_idx]
            window_labels     = self.labels[start_idx:end_idx]
            y_win             = self._majority_label(window_labels)

            X.append(window_data)
            y.append(y_win)
            times.append((start_idx / self.sfreq, end_idx / self.sfreq))

        return np.asarray(X), np.asarray(y), np.asarray(times)


class ArrayFilter(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq, lfreq, hfreq):
        self.sfreq            = sfreq
        self.lfreq           = lfreq
        self.hfreq           = hfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        b, a                 = butter(N=4, Wn=[self.lfreq, self.hfreq], btype="bandpass", fs=self.sfreq)
        X_filt               = lfilter(b, a, X, axis=-1)
        return np.asarray(X_filt)


class PseudoOnlineEvaluation:
    """
    Within-session pseudo-online only.

    - Window time-order is preserved.
    - Train/test split is temporal:
        train = first ratio windows
        test  = remaining windows

    Training behavior:
      two_stage=True:
        1) idle detector: idle vs non-idle (uses all train windows)
        2) task classifier: tasks-only (uses only y!=REST_LABEL train windows)
      two_stage=False:
        trains BOTH modes for comparison:
        - train_mode="all"       : idle + tasks (uses all train windows)
        - train_mode="tasks_only": only tasks  (uses only y!=REST_LABEL train windows)

    Output CSV has window_start/window_end + y_true + y_pred for plotting/verification.
    """

    def __init__(
        self,
        dataset,
        class_pipelines,
        wsize,
        wstep,
        subjects,
        idle_pipelines=None,
        two_stage=True,
        ratio=0.7,
        idle_threshold=0.6,
        task_threshold=0.6,
        no_run=False,
        out_prefix="results",
    ):
        self.dataset          = dataset
        self.class_pipelines  = class_pipelines
        self.idle_pipelines   = idle_pipelines
        self.two_stage        = two_stage

        self.wsize            = wsize
        self.wstep            = wstep
        self.subjects         = subjects
        self.ratio            = ratio

        self.idle_threshold   = idle_threshold
        self.task_threshold   = task_threshold
        self.no_run           = no_run
        self.out_prefix       = out_prefix

        self.results_         = []
        self.model_results_   = []

        if self.two_stage and self.idle_pipelines is None:
            raise AttributeError("two_stage=True requires idle_pipelines")

        self.dataset_name     = getattr(dataset, "code", dataset.__class__.__name__)

    @staticmethod
    def _raw_concat(raw_list):
        if len(raw_list) == 0:
            raise ValueError("Raw list is empty.")

        if len(raw_list) == 1:
            return mne.concatenate_raws(raw_list[0]) if isinstance(raw_list[0], list) else raw_list[0]

        return mne.concatenate_raws(raw_list)

    @staticmethod
    def _mask_tasks(y):
        return y != REST_LABEL

    def _append_result(
        self,
        *,
        subject,
        session,
        window,
        window_start,
        window_end,
        train_mode,
        idle_pipeline,
        task_pipeline,
        is_idle,
        y_true,
        y_pred,
        idle_proba,
        task_proba,
        t_idle_detect,
        t_task_predict,
        t_predict,
    ):
        self.results_.append(
            dict(
                dataset        = self.dataset_name,
                subject        = subject,
                session        = session,
                train_mode     = train_mode,          # "two_stage" | "all" | "tasks_only"
                idle_pipeline  = idle_pipeline,        # str or None
                task_pipeline  = task_pipeline,        # str or None
                window         = window,
                window_start   = float(window_start),
                window_end     = float(window_end),
                is_idle        = int(is_idle),
                y_true         = int(y_true),
                y_pred         = int(y_pred),
                correct        = bool(y_pred == y_true),
                idle_proba     = None if idle_proba is None else float(idle_proba),
                task_proba     = None if task_proba is None else float(task_proba),
                t_idle_detect  = None if t_idle_detect is None else float(t_idle_detect),
                t_task_predict = None if t_task_predict is None else float(t_task_predict),
                t_predict      = None if t_predict is None else float(t_predict),
            )
        )

    def _log_model_train(self, *, train_mode, pipeline_kind, pipeline_name, t_train):
        self.model_results_.append(
            dict(
                dataset       = self.dataset_name,
                train_mode    = train_mode,            # "two_stage" | "all" | "tasks_only"
                pipeline_kind = pipeline_kind,          # "idle" | "task"
                pipeline      = pipeline_name,
                t_train       = float(t_train),
            )
        )

    # ------------------------ TRAIN ------------------------ #
    def _idle_train(self, X_train, y_train):
        y_idle               = (y_train == REST_LABEL).astype(int)
        for name, pipe in self.idle_pipelines.items():
            t0               = time.perf_counter()
            pipe.fit(X_train, y_idle)
            t1               = time.perf_counter()
            self._log_model_train(train_mode="two_stage", pipeline_kind="idle", pipeline_name=name, t_train=(t1 - t0))
        return self

    def _task_train(self, X_train, y_train, *, train_mode):
        if train_mode == "tasks_only":
            mask             = self._mask_tasks(y_train)
            X_use            = X_train[mask]
            y_use            = y_train[mask]
        elif train_mode == "all":
            X_use            = X_train
            y_use            = y_train
        else:
            raise ValueError(f"Invalid train_mode: {train_mode}")

        for name, pipe in self.class_pipelines.items():
            t0               = time.perf_counter()
            pipe.fit(X_use, y_use)
            t1               = time.perf_counter()
            self._log_model_train(train_mode=train_mode, pipeline_kind="task", pipeline_name=name, t_train=(t1 - t0))
        return self

    # ------------------------ TEST ------------------------ #
    def _process_windows_two_stage(self, subject, session, X_test, y_test, times_test):
        for w in range(len(X_test)):
            win_start, win_end  = times_test[w]
            xw                  = np.asarray([X_test[w]])
            y_true              = y_test[w]

            for idle_name, idle_pipe in self.idle_pipelines.items():
                t0              = time.perf_counter()
                idle_proba       = float(idle_pipe.predict_proba(xw)[0, 0])
                is_idle          = idle_proba >= self.idle_threshold
                t1              = time.perf_counter()
                t_idle           = t1 - t0

                if is_idle:
                    self._append_result(
                        subject        = subject,
                        session        = session,
                        window         = w,
                        window_start   = win_start,
                        window_end     = win_end,
                        train_mode     = "two_stage",
                        idle_pipeline  = idle_name,
                        task_pipeline  = None,
                        is_idle        = True,
                        y_true         = y_true,
                        y_pred         = REST_LABEL,
                        idle_proba     = idle_proba,
                        task_proba     = None,
                        t_idle_detect  = t_idle,
                        t_task_predict = 0.0,
                        t_predict      = t_idle,
                    )
                    continue

                # not idle -> task prediction
                for task_name, task_pipe in self.class_pipelines.items():
                    t2          = time.perf_counter()
                    probs       = task_pipe.predict_proba(xw)[0]
                    task_proba  = float(np.max(probs))

                    y_pred      = REJECT_LABEL if task_proba < self.task_threshold else int(np.argmax(probs))
                    t3          = time.perf_counter()
                    t_task      = t3 - t2

                    self._append_result(
                        subject        = subject,
                        session        = session,
                        window         = w,
                        window_start   = win_start,
                        window_end     = win_end,
                        train_mode     = "two_stage",
                        idle_pipeline  = idle_name,
                        task_pipeline  = task_name,
                        is_idle        = False,
                        y_true         = y_true,
                        y_pred         = y_pred,
                        idle_proba     = idle_proba,
                        task_proba     = task_proba,
                        t_idle_detect  = t_idle,
                        t_task_predict = t_task,
                        t_predict      = (t_idle + t_task),
                    )

    def _process_windows_one_stage(self, subject, session, X_test, y_test, times_test, *, train_mode):
        for w in range(len(X_test)):
            win_start, win_end  = times_test[w]
            xw                  = np.asarray([X_test[w]])
            y_true              = y_test[w]

            for name, pipe in self.class_pipelines.items():
                t0          = time.perf_counter()
                probs       = pipe.predict_proba(xw)[0]
                task_proba  = float(np.max(probs))
                idle_proba  = float(probs[0]) if len(probs) else None

                y_pred      = REJECT_LABEL if task_proba < self.task_threshold else int(np.argmax(probs))
                is_idle     = (y_pred == REST_LABEL)

                t1          = time.perf_counter()
                t_pred      = t1 - t0

                self._append_result(
                    subject        = subject,
                    session        = session,
                    window         = w,
                    window_start   = win_start,
                    window_end     = win_end,
                    train_mode     = train_mode,      # "all" or "tasks_only"
                    idle_pipeline  = name,
                    task_pipeline  = name,
                    is_idle        = is_idle,
                    y_true         = y_true,
                    y_pred         = y_pred,
                    idle_proba     = idle_proba,
                    task_proba     = task_proba,
                    t_idle_detect  = (t_pred if is_idle else None),
                    t_task_predict = t_pred,
                    t_predict      = t_pred,
                )

    # ------------------------ MAIN ------------------------ #
    def evaluate(self):
        for subject in self.subjects:
            if subject not in self.dataset.subject_list:
                raise ValueError(f"Invalid subject index: {subject}")

            pre                 = self.dataset.get_data(subjects=[subject])
            raws_dict           = {}
            session_keys        = []

            # collect session raws
            for _, runs in pre.items():
                for sess, dicts in runs.items():
                    session_keys.append(sess)
                    raws_dict[sess] = []
                    for _, data in dicts.items():
                        raws_dict[sess].append(data)

            for sess in session_keys:
                raw             = self._raw_concat(raws_dict[sess])
                events, event_ids = mne.events_from_annotations(raw)

                wgen            = PseudoOnlineWindow(
                    raw         = raw,
                    events      = events,
                    interval    = self.dataset.interval,
                    task_ids    = event_ids,
                    window_size = self.wsize,
                    window_step = self.wstep,
                )

                X, y, times     = wgen.generate_windows()
                idx_split       = int(len(X) * self.ratio)

                X_train, y_train = X[:idx_split], y[:idx_split]
                X_test,  y_test  = X[idx_split:], y[idx_split:]
                times_test       = times[idx_split:]

                if self.no_run:
                    return X_train, y_train, X_test, y_test, times_test

                if self.two_stage:
                    self._idle_train(X_train, y_train)
                    self._task_train(X_train, y_train, train_mode="tasks_only")
                    self._process_windows_two_stage(subject, sess, X_test, y_test, times_test)
                else:
                    # Train and test BOTH training schemes:
                    #   - "all"       : idle + tasks
                    #   - "tasks_only": tasks only
                    for train_mode in ("all", "tasks_only"):
                        self._task_train(X_train, y_train, train_mode=train_mode)
                        self._process_windows_one_stage(subject, sess, X_test, y_test, times_test, train_mode=train_mode)

            # one save per subject (all sessions)
            if len(self.results_):
                pd.DataFrame(self.results_).to_csv(f"{self.out_prefix}-S{subject}.csv", index=False)
                self.results_ = []

            if len(self.model_results_):
                pd.DataFrame(self.model_results_).to_csv(f"{self.out_prefix}-models-S{subject}.csv", index=False)
                self.model_results_ = []

        return self
