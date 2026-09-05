import joblib
from pathlib import Path

from base.backend import ModelBackend


class MlBackend(ModelBackend):

    def get_model(self):
        return self.model

    def write(self, path: str):
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True)
        joblib.dump(self.model, path)

    def read(self, path: str):
        self.model = joblib.load(path)

    def to_internal(self, model):
        self.model = model
