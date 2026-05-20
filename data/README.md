# Data Directory

Datasets are intentionally not included in this architecture-oriented release.

If data are added locally for private experiments, use this layout:

```text
data/
|-- CALCE data/
|   `-- CALCE_Data.npy
|-- NASA data/
|   |-- B0005.mat
|   |-- B0006.mat
|   |-- B0007.mat
|   `-- B0018.mat
`-- TJU data/
    `-- Dataset_3_NCM_NCA_battery_1C.npy
```

`CALCE_Data.npy` and `Dataset_3_NCM_NCA_battery_1C.npy` should be Python dictionaries saved with `numpy.save(..., allow_pickle=True)` and loaded with `np.load(..., allow_pickle=True).item()`.
