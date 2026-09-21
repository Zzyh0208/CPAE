from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class Configuration:
    values: dict
    @classmethod
    def load(cls, path):
        return cls(yaml.safe_load(Path(path).read_text(encoding='utf-8')))
    def section(self, name): return self.values[name]
