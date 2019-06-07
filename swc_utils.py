import numpy as np 
import time
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
    
    # initialize values 
    true_pos, false_neg, false_pos = (0,0,0)
    
    # obtain true positive, false positives and false negatives
    for node in gt[:,2:5]:
        pts_within_range = [pt for pt in trace[:,2:5] if np.all(pt > (node-[dx,dy,dz])) & np.all(pt < (node+[dx,dy,dz]))]

        if len(pts_within_range) > 0:
            true_pos += 1
        else:
            false_neg += 1

    for node in trace[:,2:5]:
        pts_within_range = [pt for pt in gt[:,2:5] if np.all(pt > (node-[dx,dy,dz])) & np.all(pt < (node+[dx,dy,dz]))]

        if len(pts_within_range) == 0:
            false_pos += 1
    
    # calculate metrics         
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    f1 = 2*precision*recall / (precision + recall)
    jsc = true_pos / (true_pos + false_pos + false_neg)
                                                            
    # return metrics based on metric input value
    cases = {'all':  {'recall' : recall, 'precision' : precision, 'f1' : f1, 'jsc': jsc},
             'recall': (recall),
             'precision': (precision),
             'f1': (f1),
             'jsc': (jsc)}

    if not metric == 'all':
        return {metric : cases[metric]}
    else:
        return cases['all']

if __name__ == '__main__':
    import argparse 

    # configure argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file_path", required = True, help = "Path to swc trace file")
    ap.add_argument("-g", "--gt_file_path", required = True, help = "Path to swc of ground truth file")
    ap.add_argument("-tol", "--spatial_tol", required = False,
                    help = "spatial tolerance, as '1 1 1', with same distance measure as the swc files",
                    type = float , nargs = 3, default = [3.0,3.0,3.0])
    ap.add_argument("-met", "--metric", required = False, help = "desired output metrics",
                    default = 'all') 
    args = vars(ap.parse_args())

    # unpack list
    dx, dy, dz = args['spatial_tol']
    met = args['metric'] 

    # run metric computation and take time 
    start = time.time()
    result = geometry_metric(args['file_path'], args['gt_file_path'], dx, dy, dz, met)
    end = time.time()
    print(f'\nResults: {result}.\nTime to process: {end-start:.3f}s\n')
