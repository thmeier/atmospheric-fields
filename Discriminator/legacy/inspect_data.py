import xarray as xr
import os
from omegaconf import OmegaConf

def check_files():
    conf = OmegaConf.load("conf/config.yaml")
    data_dir = conf.data_dir
    files = [
        "graphcast_6steps_1.5deg_2020-01-01_2020-12-31.nc",
        "sphericalcnn_6steps_1.5deg_2020-01-01_2020-12-31.nc",
        "pangu_6steps_1.5deg_2020-01-01_2020-12-31.nc",
        "fuxi_6steps_1.5deg_2020-01-01_2020-12-31.nc"
    ]
    
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        try:
            ds = xr.open_dataset(path, decode_times=False)
            print(f"\nFile: {f}")
            if "prediction_timedelta" in ds.coords:
                pt = ds.prediction_timedelta
                print(f"  prediction_timedelta dtype: {pt.dtype}")
                print(f"  prediction_timedelta values: {pt.values[:3]}")
                if "units" in pt.attrs:
                    print(f"  prediction_timedelta units: {pt.attrs['units']}")
            else:
                print("  prediction_timedelta not found in coords")
        except Exception as e:
            print(f"  Error reading {f}: {e}")

if __name__ == "__main__":
    check_files()
