# ASL_coil
See progress in [gantt chart](https://prod.teamgantt.com/gantt/schedule/?ids=2329320#ids=2329320&user=&custom=&company=&hide_completed=false&date_filter=&color_filter=).

# Dependancies
Install shimmingtoolbox package:
```
$ pip install shimmingtoolbox@git+https://github.com/shimming-toolbox/shimming-toolbox
```
or see [the repository here](https://github.com/shimming-toolbox/shimming-toolbox).

Move all data folders directly into the repository, if not there. 

# Usage
Run `ASL_optimize_shim.py`. To update neck coil generation, change `biot_savart_neck.py`, and delete or move the existing neck coil file. To change plotting, modify `compare_shim.py`. Modify `ASL_optimize_shim.py` to change what coils and subjects are used. 

# For selecting masks for .mat files
Create a .mat file with the variable accesible at the top (a mat file with just the taret variable is ideal). Give the in/out file and variable names in `matlab_mask.py`, and run. View any 3D matrix M with `imagesc3D(M)`. 
