"""Output paths and atomic metadata writes; no automatic reports or QA folders."""
from dataclasses import dataclass
import json
from pathlib import Path
from simulation.study import valid_name


def dump(path, value):
    path=Path(path)
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    temporary=path.with_name(path.name+'.tmp')
    try:
        temporary.write_text(payload,encoding='utf-8')
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class OutputLayout:
    root: Path

    @property
    def raw(self):
        return self.root/'raw'

    @property
    def final(self):
        return self.root/'final'

    @property
    def qa(self):
        return self.root/'_work/qa'

    def create_case(self,name=None):
        path=self.raw/valid_name(name) if name is not None else self.raw
        path.mkdir(parents=True,exist_ok=False)
        return path

    def prepare_figures(self,overwrite=False,qa=False):
        self.final.mkdir(parents=True,exist_ok=overwrite)
        if qa:
            self.qa.mkdir(parents=True,exist_ok=True)


def input_raw(path):
    path=Path(path)
    return path/'raw' if (path/'raw').is_dir() else path
