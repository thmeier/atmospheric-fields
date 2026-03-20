# Metric for Realism of Atmospheric Fields

Project work for the course ETH AI Center Projects in Machine Learning Research 2026 at ETH Zurich.

## Setup with conda

Create a new conda environment using python 3.13, for example:

```bash
$ conda create -n pmlr python=3.13 -y && conda activate pmlr
```

Populate the new environment with the list of packages found in `./requirements.txt`

```bash
$ conda install --file requirements.txt
```

In case the installation from the requirements.txt fails, try adding the conda-forge repo.

```bash
$ conda config --add channels conda-forge
```

## Get Initial Test Data

To download an initial batch of ERA5 test data, have a look at the file `start_download_era5.sh`.
Each of the environment variables should be self explanatory, the most interesting ones are

```
         SOURCE : google cloud bucket link, this implies the resolution of the data
     TIME_START : in YYYY-MM-DD format, observations will start from this date
       TIME_END : in YYYY-MM-DD format, observations will end on this date
      VARIABLES : which variables to select from the observations, if empty, select all
OUTPUT_FILENAME : filename for the data, mind the .nc file extension
     OUTPUT_DIR : where the data will be saved to, mind that /cluster/courses/pmlr/teams/team07 is shared
```
