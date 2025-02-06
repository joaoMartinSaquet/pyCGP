from pycgp import CGP, CGPES
from pycgp.evaluators import SREvaluator
from pycgp.cgpfunctions import *
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import sympy as sp


def build_funcLib():
    return [CGP.CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGP.CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGP.CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGP.CGPFunc(f_sin, 'sin', 1, 0, 'sin'),
            CGP.CGPFunc(f_cos, 'cos', 1, 0, 'cos'),
            ]
def fit_me(x):
    return np.sin(x*x*x) + x**2

def evolve(folder_name, col=10, row=1, nb_ind=8, mutation_rate_nodes=0.1, mutation_rate_outputs=0.2,
              n_cpus=1, n_it=1000, genome=None):
    

    x_train = np.random.uniform(-5, 5, 100)
    sig = 0.01
    y_train = fit_me(x_train) + np.random.normal(0, sig, 100)

    e = SREvaluator(x_train=x_train, y_train=y_train, loss='mse')
    
    library = build_funcLib()
    if genome is None:
        cgpFather = CGP.random(1, 1, col, row, library, 1, False, const_min=0, const_max=1, input_shape=x_train.shape, dtype='float')
    else:
        cgpFather = CGP.load_from_file(genome, library)
    print(cgpFather.genome)
    es = CGPES(nb_ind, mutation_rate_nodes, mutation_rate_outputs, cgpFather, e, folder_name, n_cpus)
    es.run(n_it)

    # for i in range(10):
    #     print(es.father.genome)
    #     print(e.evaluate(es.father, 0))

    f_str = es.father.to_function_string(['x'], ['y'])

    



    es.father.to_dot('best.dot', ['x'], ['y'])
    os.system('dot -Tpdf ' + 'best.dot' + ' -o ' + 'best.pdf')

    x = np.linspace(-2, 5, 150)
    es.father.input_shape = x.shape
    es.father.graph_created = False
    plt.plot(x_train, y_train, 'rx', label='train')
    plt.plot(x, es.father.run(x)[0], 'b', label='res')
    plt.savefig("graph.png")
    print("HOF not simplified : ", f_str[-1])
    print("HOF func simplified : ", sp.simplify(f_str[-1]))

# def load(file_name):
#     print('loading ' + file_name)
#     library = build_funcLib()
#     c = CGP.load_from_file(file_name, library)
#     print(e.evaluate(c, 0, displayTrace=True))
   
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
        evolve('test')
    if len(sys.argv)==2:
        print('Starting evolution from random genome')
        evolve(sys.argv[1])
    elif len(sys.argv)==3:
        print('Starting evolution from genome saved in ', sys.argv[2])
        evo(sys.argv[1], genome=sys.argv[2])