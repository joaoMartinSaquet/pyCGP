from pycgp import CGP, CGPES
from pycgp.evaluators import SREvaluator
from pycgp.cgpfunctions import *
import numpy as np
import sys
import os



def build_funcLib():
    return [CGP.CGPFunc(f_sum, 'sum', 2, 0),
            CGP.CGPFunc(f_aminus, 'aminus', 2, 0),
            CGP.CGPFunc(f_mult, 'mult', 2, 0),
            CGP.CGPFunc(f_exp, 'exp', 2, 0),
            CGP.CGPFunc(f_abs, 'abs', 1, 0),
            CGP.CGPFunc(f_sqrt, 'sqrt', 1, 0),
            CGP.CGPFunc(f_sqrtxy, 'sqrtxy', 2, 0),
            CGP.CGPFunc(f_squared, 'squared', 1, 0),
            CGP.CGPFunc(f_pow, 'pow', 2, 0),
            CGP.CGPFunc(f_one, 'one', 0, 0),
            CGP.CGPFunc(f_zero, 'zero', 0, 0),
            CGP.CGPFunc(f_const, 'const', 0, 1),
            # CGP.CGPFunc(f_inv, 'inv', 1, 0),
            # CGP.CGPFunc(f_gt, 'gt', 2, 0),
            CGP.CGPFunc(f_asin, 'asin', 1, 0),
            CGP.CGPFunc(f_acos, 'acos', 1, 0),
            CGP.CGPFunc(f_atan, 'atan', 1, 0),
            CGP.CGPFunc(f_min, 'min', 2, 0),
            CGP.CGPFunc(f_max, 'max', 2, 0),
            CGP.CGPFunc(f_round, 'round', 1, 0),
            CGP.CGPFunc(f_floor, 'floor', 1, 0),
            CGP.CGPFunc(f_ceil, 'ceil', 1, 0)
            ]


def evolveSin(folder_name, col=5, row=1, nb_ind=4, mutation_rate_nodes=0.1, mutation_rate_outputs=0.3,
              n_cpus=1, n_it=100, genome=None):
    

    x_train = np.random.uniform(0, 2*np.pi, 100)
    sig = 0.001 
    y_train = np.sin(x_train) + np.random.normal(0, sig, x_train.shape)

    e = SREvaluator(x_train=x_train, y_train=y_train)
    
    library = build_funcLib()
    if genome is None:
        cgpFather = CGP.random(1, 1, col, row, library, 1.0, False, const_min=0, const_max=1, input_shape=x_train.shape, dtype='float')
    else:
        cgpFather = CGP.load_from_file(genome, library)
    print(cgpFather.genome)
    es = CGPES(nb_ind, mutation_rate_nodes, mutation_rate_outputs, cgpFather, e, folder_name, n_cpus)
    es.run(n_it)

    for i in range(10):
        print(es.father.genome)
        print(e.evaluate(es.father, 0, False))

    es.father.to_function_string(['x'], ['y'])
#    es.father.to_dot(folder_name+'/best.dot', ['x'], ['y'])
#    os.system('dot -Tpdf ' + folder_name+'/best.dot' + ' -o ' + folder_name+'/best.pdf')


def load(file_name):
    print('loading ' + file_name)
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    e = SinEvaluator()
    print(e.evaluate(c, 0, displayTrace=True))
   
def toDot(file_name, out_name):
    print('Exporting ' + file_name + ' in dot ' + out_name + '.dot')
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    c.to_dot(out_name+'.dot', ['x'], ['y'])
    print('Converting dot file into pdf in ' + out_name + '.pdf')
    os.system('dot -Tpdf ' + out_name + '.dot' + ' -o ' + out_name + '.pdf')

def displayFunctions(file_name):
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    c.to_function_string(['x'], ['y'])


if __name__ == '__main__':
    print (sys.argv)
    if len(sys.argv) == 1:
        evolveSin('test')
    if len(sys.argv)==2:
        print('Starting evolution from random genome')
        evolveSin(sys.argv[1])
    elif len(sys.argv)==3:
        print('Starting evolution from genome saved in ', sys.argv[2])
        evo(sys.argv[1], genome=sys.argv[2])