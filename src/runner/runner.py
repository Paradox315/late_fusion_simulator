from typing import Callable, Dict, Tuple

import numpy as np
import torch

from src.models.frame import Frame
import json as js


class Simulator:
    def __init__(self, record_length: int, initial_frame: Frame):
        self.frames = [initial_frame]
        self.record_length = record_length
        for i in range(record_length - 1):
            next_frame = next(self.frames[-1])
            if next_frame is None:
                break
            self.frames.append(next_frame)
        self.results = {}

    def run(
        self,
        fuse_method: Callable[[np.ndarray, np.ndarray], Tuple],
        visualize=False,
    ):
        self.results = {}
        for frame in self.frames:
            if visualize:
                frame.visualize()
            preds = frame.predict(fuse_method)
            self._update_results(preds)
        return self.results

    def get_dataset(self):
        results = []
        for frame in self.frames:
            data = frame.generate_data()
            results.extend(data)
        train_idx = int(len(results) * 0.7)
        val_idx = int(len(results) * 0.75)
        return results[:train_idx], results[train_idx:val_idx], results[val_idx:]

    def _update_results(self, preds):
        for ego, preds in preds.items():
            if ego not in self.results:
                self.results[ego] = {}
            for cav, pred in preds.items():
                self.results[ego].setdefault(cav, []).append(pred)

    def get_records_length(self):
        return sum([len(v) for v in self.results.values()])

    def save_simulation_results(self, format="console", path=None):
        if format == "console":
            print(js.dumps(self.results, indent=4, ensure_ascii=False))
        elif format == "json":
            if path is None:
                path = "../data/simulation_results.json"
            with open(path, "w") as f:
                js.dump(self.results, f)
        elif format == "csv":
            import pandas as pd

            if path is None:
                path = "../data/simulation_results.csv"
            df = pd.DataFrame(self.results)
            df.to_csv(path)
        else:
            raise ValueError("Unsupported format")
