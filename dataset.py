import torch
from torch.utils.data import Dataset
import numpy as np
from netCDF4 import Dataset as NetCDFDataset

class AtmosphereDataset(Dataset):
    def __init__(
        self,
        data_path,
        split="train",
        train_ratio=0.8,
        stats=None,
        lazy=False,
        stats_chunk_size=256,
    ):
        """
        Loads ERA5 data. 
        If lazy=False, eagerly loads into memory.
        If lazy=True, keeps xarray pointer and loads slice-by-slice per __getitem__.
        """
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.lazy = lazy
        self.stats_chunk_size = stats_chunk_size
        self.ds = None
        
        self.vars = [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure"
        ]

        with NetCDFDataset(data_path, mode="r") as ds:
            self.total_samples = len(ds.dimensions["time"])

        split_idx = int(self.total_samples * train_ratio)
        
        if split == "train":
            self.start_idx = 0
            self.end_idx = split_idx
        elif split == "val":
            self.start_idx = split_idx
            self.end_idx = self.total_samples
        else:
            self.start_idx = 0
            self.end_idx = self.total_samples
            
        self.n_samples = self.end_idx - self.start_idx
        print(f"[{split}] Dataset initialized with {self.n_samples} samples. Lazy={lazy}")

        # Compute or apply normalization stats
        if stats is None:
            if split != "train":
                print("Warning: Creating val/test dataset without train stats!")
            print(f"[{split}] Computing normalization stats...")
            self.mean, self.std = self._compute_stats()
        else:
            self.mean = np.asarray(stats[0], dtype=np.float32)
            self.std = np.asarray(stats[1], dtype=np.float32)

        self.mean_1d = self.mean.reshape(-1).astype(np.float32, copy=False)
        self.std_1d = self.std.reshape(-1).astype(np.float32, copy=False)
        self.std_1d[self.std_1d == 0] = 1.0

        if not self.lazy:
            print(f"[{split}] Eagerly loading {self.n_samples} samples into memory...")
            data = self._read_time_slice(self.start_idx, self.end_idx)
            data = self._normalize_and_pad(data)
            self.data_tensor = torch.from_numpy(data)

    def _get_dataset(self):
        if self.ds is None:
            self.ds = NetCDFDataset(self.data_path, mode="r")
        return self.ds

    def _read_time_slice(self, start_idx, end_idx):
        ds = self._get_dataset()
        arrays = []
        for var_name in self.vars:
            # NetCDF variable layout is (time, longitude, latitude); transpose to (time, latitude, longitude).
            arr = np.asarray(ds.variables[var_name][start_idx:end_idx, :, :], dtype=np.float32)
            arr = np.transpose(arr, (0, 2, 1))
            arrays.append(arr)
        return np.stack(arrays, axis=1).astype(np.float32, copy=False)

    def _normalize_and_pad(self, data):
        data = np.asarray(data, dtype=np.float32)
        data = (data - self.mean) / self.std
        data = np.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (8, 8)), mode="wrap")
        data = np.pad(data, pad_width=((0, 0), (0, 0), (4, 3), (0, 0)), mode="constant", constant_values=0)
        return np.asarray(data, dtype=np.float32)

    def _compute_stats(self):
        total_count = 0
        total_sum = np.zeros(4, dtype=np.float64)
        total_sumsq = np.zeros(4, dtype=np.float64)

        for chunk_start in range(self.start_idx, self.end_idx, self.stats_chunk_size):
            chunk_end = min(chunk_start + self.stats_chunk_size, self.end_idx)
            chunk = self._read_time_slice(chunk_start, chunk_end)
            total_sum += chunk.sum(axis=(0, 2, 3), dtype=np.float64)
            total_sumsq += np.square(chunk, dtype=np.float32).sum(axis=(0, 2, 3), dtype=np.float64)
            total_count += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]

        mean = total_sum / total_count
        var = (total_sumsq / total_count) - np.square(mean)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)

        mean = mean.reshape(1, 4, 1, 1).astype(np.float32)
        std = std.reshape(1, 4, 1, 1).astype(np.float32)
        return mean, std

    def get_stats(self):
        return self.mean, self.std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if not self.lazy:
            return self.data_tensor[idx]

        actual_idx = self.start_idx + idx
        sample = self._read_time_slice(actual_idx, actual_idx + 1)
        sample = self._normalize_and_pad(sample)[0]
        return torch.from_numpy(sample)

if __name__ == "__main__":
    from pathlib import Path
    data_path = Path("data/test_data_local.nc")
    if data_path.exists():
        ds_lazy = AtmosphereDataset(data_path, split="train", lazy=True)
        print("Lazy dataloader shape:", ds_lazy[0].shape)
