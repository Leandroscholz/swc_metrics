import numpy as np 
import time
import functools
import multiprocessing 

def geometry_metric(trace, ground_truth, dx = 3.0, dy = 3.0, dz = 3.0):
    trace = np.loadtxt(trace)
    gt = np.loadtxt(ground_truth)
    
    pool = multiprocessing.Pool(processes = 8)

    true_pos = sum(pool.map(functools.partial(true_positives,traces=trace,tol=tuple((dx,dy,dz))),gt[:,2:5]))
    false_pos = sum(pool.map(functools.partial(false_positives,ground_truth=gt,tol=tuple((dx,dy,dz))),trace[:,2:5]))

    false_neg = len(gt)-true_pos

    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    
    return recall, precision 

def true_positives(node,traces,tol):
    dx, dy, dz = tol
    pts_within_range = [pt for pt in traces[:,2:5] if np.all(pt > (node-[dx,dy,dz])) & np.all(pt < (node+[dx,dy,dz]))]

    if len(pts_within_range) > 0:
        return 1
    else:
        return 0

def false_positives(node,ground_truth,tol):
    dx, dy, dz = tol
    pts_within_range = [pt for pt in ground_truth[:,2:5] if np.all(pt > (node-[dx,dy,dz])) & np.all(pt < (node+[dx,dy,dz]))]

    if len(pts_within_range) > 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    start = time.time()
    recall, precision = geometry_metric('OP_1_app1_fixed.swc','OP_1.swc')
    end = time.time()
    print(f'\nRecall: {recall:.3f} Precision: {precision:.3f}. Time to process: {end-start:.3f}s\n')
