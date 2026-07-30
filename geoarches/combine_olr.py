from glob import glob

import xarray as xr

pattern = "data/era5_1x1_daily/full/*.nc"

files = glob(pattern, recursive=True)
olr_files = [f for f in files if "OLR" in f]
olr_files.sort()
era_files = [f for f in files if "OLR" not in f]
era_files.sort()
era_files = era_files[:-1]
for olr, era in zip(olr_files, era_files):
    print(olr, era)
    print("----------------------------------------")
    era_ds = xr.open_dataset(era)
    olr_ds = xr.open_dataset(olr)
    olr_ds = olr_ds.rename_vars({"ttr": "top_net_longwave_radiation"})

    era_ds = xr.merge([era_ds, olr_ds])
    era_ds.to_netcdf(era)
    olr_ds.close()
    era_ds.close()


r"""file_tuples = [files[i:i+2] for i in range(0, len(files)-1, 2)]

for file_tuple in file_tuples:
    ds = xr.open_mfdataset(file_tuple)
    year = re.findall(pattern='\d{4}', string=file_tuple[0])[0]
    print(f"####### {year} ########")
    print(file_tuple)

    ds.to_netcdf(f"data/era5_1x1_daily/full/era5_OLR_{year}.nc")"""
