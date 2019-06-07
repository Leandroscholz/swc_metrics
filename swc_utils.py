import numpy as np 
import time
import multiprocessing 

def geometry_metric(trace, ground_truth, dx = 3.0, dy = 3.0, dz = 3.0):
    trace = np.loadtxt(trace)
    gt = np.loadtxt(ground_truth)
   
    true_pos = 0
    false_neg = 0
    false_pos = 0

    for node in gt[:,2:5]:
        pts_within_range = [pt for pt in trace[:,2:5] if np.all(pt > (node-[dx,dy,dz])) & np.all(pt < (node+[dx,dy,dz]))]

        if len(pts_within_range) > 0:
            true_pos += 1
        else:
            false_neg += 1

    recall = true_pos / (true_pos + false_neg)
    
    for node in trace[:,2:5]:
        pts_within_range = [pt for pt in gt[:,2:5] if np.all(pt > (node-[dx,dy,dz])) & np.all(pt < (node+[dx,dy,dz]))]

        if len(pts_within_range) == 0:
            false_pos += 1

    precision = true_pos / (true_pos + false_pos)

    return recall, precision 

if __name__ == '__main__':
    start = time.time()
    recall, precision = geometry_metric('OP_1_app1_fixed.swc','OP_1.swc')
    end = time.time()
    print(f'\nRecall: {recall:.3f} Precision: {precision:.3f}. Time to process: {end-start:.3f}s\n')

