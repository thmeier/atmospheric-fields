#!/bin/bash
# resolution of 1.5 degree
# start time of observations, YYYY-MM-DD format
#TIME_START="2004-01-01"
# end time of observations, YYYY-MM-DD format
#TIME_END="2023-12-31"

#SBATCH --time=02:00
#SBATCH --account=pmlr_jobs

#TIME_START2="2018-01-01"
#TIME_END2="2018-12-31"
TIME_START1="2020-01-01"
TIME_END1="2020-12-31"
#bash download_era5_args.sh era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr $TIME_START1 $TIME_END1 era5-gt
#bash download_era5_args.sh graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr $TIME_START2 $TIME_END2 graphcast
#bash download_era5_args.sh graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr $TIME_START1 $TIME_END1 graphcast
#bash download_era5_args.sh pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr $TIME_START1 $TIME_END1 pangu
#bash download_era5_args.sh pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr $TIME_START2 $TIME_END2 pangu
#bash download_era5_args.sh era5-forecasts/2020-240x121_equiangular_with_poles_conservative.zarr $TIME_START1 $TIME_END1 era5_forecast
#bash download_era5_args.sh hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr $TIME_START1 $TIME_END1 ifs_hres
#bash download_era5_args.sh keisler/2020-240x121_equiangular_with_poles_conservative.zarr $TIME_START1 $TIME_END1 keisler
bash download_era5_args.sh sphericalcnn/2020-240x121_equiangular_with_poles.zarr $TIME_START1 $TIME_END1 sphericalcnn
#bash download_era5_args.sh neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr $TIME_START1 $TIME_END1 neuralgcm
#bash download_era5_args.sh fuxi/2020-240x121_equiangular_with_poles_conservative.zarr $TIME_START1 $TIME_END1 fuxi
#bash download_era5_args.sh hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr $TIME_START2 $TIME_END2 ifs_hres
