# Prediction of the runoff and sediment discharge evolutions at the Yellow River Delta by considering the basin irrigation

This is the official Python implementation of the paper:

*Prediction of the runoff and sediment discharge evolutions at the Yellow River Delta by considering the basin irrigation*<>

by Zhen Yang et al.

## Requirements 

*1.* Download current version of the repository.

```
git clone https://github.com/Mor-yz/Prediction-of-the-runoff-and-sediment-discharge-evolutions-at-the-Yellow-River-Delta.git
```

*2.* Install the dependencies in the `requirements.txt` file.<br>

## Running the code

SAC variables are constructed in `sac.py`.

QBSO code for feature selection is in the folder `qbso-fs-master`. The original QBSO code is in

```
git clone https://github.com/amineremache/qbso-fs.git
```

We add RR regressor in `fs_problem.py` file. The codes in this part are explained in more detail in the depository above.

RR after feature selection is in `rr_irr.py`.

Residual correction is in `rc.py`, which includes EMD and GRU.

## Data

All data in the process is in the folder `data`.

In the folder `process data`, the last column of excels similar to `y1_linear.csv` represents the dependent variable, and the rest represents the independent variable, and they can be directly put into QBSO to run.

The nc files of IWU are shown in `IWUEN`.

The runoff and sediment discharge observed in Lijin station are showed in `lijin.xlsx`, and data after logarithm are in `log.xlsx`.

The final prediction results are showed in `final_runoff.xlsx` and `final_sediment.xlsx`.

coordinates of high impact points are showed in `coordinate_runoff.xls` and `coordinate_sediment.xls`.

