import enum
import numpy as np
import pandas as pd
import pathlib
import torch
import typing as t

from collections.abc import Iterable
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from pathlib import Path



class Modes(str, enum.Enum):
    ADJACENT = "adjacent"
    CONTOUR = "contour"
    DEPTH = "depth"
    GRADANGLE = "gradangle"
    MEDIAN = "median"
    LAPLACIAN = "laplacian"


class CowDataset(Dataset):
    def __init__(
        self,
        root: pathlib.Path | str,
        labels_filename: pathlib.Path | str,
        modes: Modes | t.Iterable[Modes] = "gradangle",
        resize_shape: tuple[int, int] = (512, 512),
        interpolation_mode: InterpolationMode = InterpolationMode.BILINEAR,
        antilias: bool = False,
        dtype=torch.float64,
    ):
        self.root = root
        self.labels_filename = labels_filename
        self.modes = CowDataset._resolve_mode_kind(modes)
        self.resizer = transforms.Resize(
            resize_shape, interpolation_mode, antialias=antilias
        )

        # TODO: This will need to be re-implemented for when we transition to a larger
        # part of the dataset since everything is currently being held in memory.
        records = []
        num_iters = 0
        labels_csv = pd.read_csv(self.labels_filename)

        for filename in root.glob("*.npz"):
            file_rows = {}
            stem = CowDataset._process_filename_stem(filename.stem)
            record_base = {"stem": stem, "path": filename}

            np_data = np.load(filename, allow_pickle=True)
            for mode in self.modes:
                for frame_idx, frame in enumerate(np_data[mode]):
                    bcs = labels_csv.query("id == @stem").iloc[0].bcs
                    bcs = (bcs - 100) / 100.0  # Transform to standard 0-4 range.

                    record = record_base.copy()
                    record.update(
                        {
                            "frame": frame_idx,
                            "mode": torch.from_numpy(frame).type(dtype),
                            "bcs": bcs,
                        }
                    )
                    records.append(record)
                    num_iters += 1

        self._data = pd.DataFrame.from_records(records)
        self._data = self._data.dropna(how="any")
        self._data = self._data.reset_index()

    def __str__(self) -> str:
        return f"CowDataset(root={self.root}, modes={self.modes})"

    def __getitem__(self, batch_idx: int) -> tuple[dict[Modes, torch.Tensor], float]:
        """Getter method for accessing data by index.

        Args:
            batch_idx (int): The index of the datum to return.

        Returns:
            tuple[dict[str, torch.Tensor], float]: Pair of inputs and targets. Note, the
            the inputs are a dictionary with the "mode" being the key (see `Modes` enum)
            and the values being the corresponding `torch.Tensor`.
        """
        row = self._data.iloc[batch_idx]
        inputs = {mode: self.resizer(row['mode'].unsqueeze(dim=0)) for mode in self.modes}
        targets = row.bcs

        return inputs, targets

    def __len__(self) -> int:
        """Returns the length of the underlying Dataframe.

        Returns:
            int: Length of dataset.
        """
        return len(self._data)

    @staticmethod
    def _process_filename_stem(stem: str) -> str:
        split_stem = stem.split("_")
        return f"{split_stem[0]}_{split_stem[1]}"

    @staticmethod
    def _resolve_mode_kind(modes: str | Modes | t.Iterable[Modes]) -> list[Modes]:
        """
        Helper function that handles the type ambiguity for the `modes` argument. Simplifies
        the typing by returning a `list` of `Modes` string literals.

        Args:
            modes (str | Modes | t.Iterable[str | Modes]): Modes to be resolved.

        Raises:
            ValueError: Will be thrown in the event that an illegal value is provided in arg `modes`.

        Returns:
            list[Modes]: The modes to be used for loading in data.
        """
        match modes:
            case tuple() | list() | set():
                # May raise `ValueError` if illegal value is provided.
                return [Modes(val) for val in modes]
            case Modes():
                return [modes.value]
            case str():
                Modes(modes)  # May raise `ValueError` if illegal value is provided.
                return [modes]
            case _:
                raise ValueError


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    
    from src.const import DEFAULT_ROOT_DIR

    data = CowDataset(
        DEFAULT_ROOT_DIR,
        "../processed_bcs_labels.csv",
        modes=["depth", "median", "gradangle"],
    )



    print(data)
    print("Number of data samples:", len(data))
