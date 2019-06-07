import numpy as np 
import time
import functools
import multiprocessing 

def geometry_metric(trace, ground_truth, dx = 3.0, dy = 3.0, dz = 3.0, metric = 'all'):
    """
    geometry metric computes geometric scores of swc trace files
    against a single ground truth swc file. It can compute recall,
    precision, f1 score and the Jaccard Similarity Coefficient (JSC)
    the range of detection and the desired outputs are given by the user 
    
    inputs
        trace           file path to .swc tracing file 
        ground truth    file path to .swc ground truth file  
        dx, dy, dz      tolerance in distance measure, same as the unit 
                            of the swc file (default dx, dy, dz = 3 px)
        metric          the desired output metrics (detault = 'all')
                            options: 'all' (recall, precision, f1, jsc)
                                     'recall'
                                     'precision'
                                     'f1'
                                     'jsc'

    """

    # load files 
    trace = np.loadtxt(trace)
    gt = np.loadtxt(ground_truth)
    
    # create multiprocessing pool 
    pool = multiprocessing.Pool(processes = 8)

    # obtain true positive, false positives and false negatives with multiprocesses
    true_pos = sum(pool.map(functools.partial(true_positives,traces=trace,tol=tuple((dx,dy,dz))),gt[:,2:5]))
    false_neg = len(gt)-true_pos
    false_pos = sum(pool.map(functools.partial(false_positives,ground_truth=gt,tol=tuple((dx,dy,dz))),trace[:,2:5]))
    
    # calculate metrics 
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    f1 = 2*precision*recall / (precision + recall)
    jsc = true_pos / (true_pos + false_pos + false_neg)

    # return metrics based on metric input value
    cases = {'all':  (recall, precision, f1, jsc),
             'recall': (recall),
             'precision': (precision),
             'f1': (f1),
             'jsc': (jsc)}

    return cases[metric]

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
    recall, precision, *rest = geometry_metric('OP_1_app1_fixed.swc','OP_1.swc',metric = 'all')
    end = time.time()
    print(f'\nRecall: {recall:.3f} Precision: {precision:.3f}. Time to process: {end-start:.3f}s\n')
    
    start = time.time() 
    precision = geometry_metric('OP_1_app1_fixed.swc','OP_1.swc',metric = 'precision')
    end = time.time()
    print(f'\nPrecision: {precision:.3f}. Time to process: {end-start:.3f}s\n')
