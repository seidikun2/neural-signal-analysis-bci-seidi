import os, re, inspect, warnings, moabb
from moabb.datasets  import Stieger2021

warnings.filterwarnings("ignore")
moabb.set_log_level("info")

DATA_DIR = r"/home/seidi/Downloads/Stieger"
SUBJECT  = 1
SESSION  = 1
INTERVAL = [0.5, 2.5]


class Stieger2021Local(Stieger2021):

    def __init__(self, interval=[0, 3], sessions=None, fix_bads=True, data_dir=None):
        sig = inspect.signature(super().__init__)
        if "fix_bads" in sig.parameters:
            super().__init__(interval=interval, sessions=sessions, fix_bads=fix_bads)
        else:
            super().__init__(interval=interval, sessions=sessions)
        self.data_dir = data_dir

    def data_path(self, subject, **kwargs):
        files = []
        for f in os.listdir(self.data_dir):
            if not f.endswith(".mat"):
                continue
            m = re.match(r"S(\d+)_Session_(\d+)\.mat", f)
            if m and int(m.group(1)) == subject:
                ses = int(m.group(2))
                if self.sessions is None or ses in self.sessions:
                    files.append(os.path.join(self.data_dir, f))
        return sorted(files)


if __name__ == "__main__":

    ds = Stieger2021Local(
        interval = INTERVAL,
        sessions = [SESSION],
        data_dir = DATA_DIR
    )
    ds.subject_list = [SUBJECT]

    # --- AQUI está o ponto certo ---
    subj_dict = ds._get_single_subject_data(SUBJECT)

    # sessões podem vir como str ou int
    runs = subj_dict.get(str(SESSION), {}) or subj_dict.get(SESSION, {})

    if not runs:
        raise RuntimeError("Nenhum run encontrado para essa sessão.")

    # pega o primeiro run (suficiente para listar canais)
    raw = next(iter(runs.values()))

    ch_names = raw.copy().pick("eeg").info["ch_names"]

    print(f"\nSujeito {SUBJECT} | Sessão {SESSION}")
    print(f"Número de canais EEG: {len(ch_names)}")
    print("Canais disponíveis:")
    print(sorted(ch_names))
