# swc_metrics

this repository houses some filament tracing metric computations. It includes (initially) two modules `swc_utils.py` and `swc_utils_parallel.py`, which perform the exact same tasks with the difference that the latter uses python's `multiprocessing` library and computes metrics a bit faster than the the first module. 

dependencies: `numpy`, `time`, `functols`, `multiprocessing`

example calls: 
    using default tolerance dx, dy, dz = [3.0,3.0,3.0] and computing all metrics 
```
python3 swc_utils_parallel.py -f 'OP_1_app1_fixed.swc' -g 'OP_1.swc' 

```
    using specific tolerance and computing specific metric
```
python3 swc_utils_parallel.py -f 'OP_1_app1_fixed.swc' -g 'OP_1.swc' -tol 2.5 2.5 3 -m 'precision'
```
