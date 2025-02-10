from pycgp import CGP, CGPES
from pycgp.evaluators import SREvaluator
from pycgp.viz import draw_net
from pycgp.cgpfunctions import *
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import sympy as sp
import networkx as nx

def build_funcLib():
    return [CGP.CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGP.CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGP.CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGP.CGPFunc(f_sin, 'sin', 1, 0, 'sin'),
            CGP.CGPFunc(f_cos, 'cos', 1, 0, 'cos'),
            ]
def fit_me(x):
    return np.sin(x*x) 

def base(folder_name, col=30, row=1, nb_ind=4, mutation_rate_nodes=0.1, mutation_rate_outputs=0.2,
              n_cpus=1, n_it=500, genome=None):
    
    Ntrain = 100
    x_train = np.random.uniform(-5, 5, Ntrain)
    sig = 0.0
    y_train = fit_me(x_train) + np.random.normal(0, sig, Ntrain)
    library = build_funcLib()
    e = SREvaluator(x_train=x_train, y_train=y_train, n_inputs=1, n_outputs=1, library=library, loss='mse')
    
    best = e.evolve(nb_ind, mutation_rate_nodes, mutation_rate_outputs, n_cpus, n_it, folder_name)
    
    f_str = best.to_function_string(['x'], ['y'])

    
    best.to_dot('best.dot', ['x'], ['y'])
    os.system('dot -Tpdf ' + 'best.dot' + ' -o ' + 'best.pdf')
    print("father genome : ", best.genome)
    x = np.linspace(-5, 5, 150)
    best.input_shape = x.shape
    best.graph_created = False
    plt.plot(x_train, y_train, 'rx', label='train')
    plt.plot(x, best.run(x)[0], 'b', label='res')
    # plt.savefig("graph.png")
    plt.figure()
    G = best.netx_graph(['x'], ['y'], active=False)
    draw_net(G, y_offset=100)
    G = best.netx_graph(['x'], ['y'], active=True)
    draw_net(G, y_offset=400, node_shape='s')
    

    print("HOF not si   mplified : ", f_str[-1])   
    print("HOF func simplified : ", sp.simplify(f_str[-1]))
    plt.show()

# def load(file_name):
#     print('loading ' + file_name)
#     library = build_funcLib()
#     c = CGP.load_from_file(file_name, library)
#     print(e.evaluate(c, 0, displayTrace=True))
def roses(folder_name, col=30, row=1, nb_ind=4, mutation_rate_nodes=0.1, mutation_rate_outputs=0.2,
              n_cpus=1, n_it=500, genome=None):

    n = 200
    A = 1.995633
    B = 1.27689
    C = 8
    r = np.linspace(0, 1, n)
    th = np.linspace(-2, 20*np.pi, n)
    x = r*np.cos(th)
    y = r*np.sin(th)
    [R ,THETA] = np.meshgrid(r, th)
    petal_number = 3.6 

    x = 1 - 0.5*( 1.25* (1-np.mod(petal_number*THETA, 2*np.pi)/np.pi)**2  - 0.25)**2
    phi = (np.pi/2)*np.exp(-THETA/(C*np.pi))
    y = A*(R**2)*(B*R-1)**2*np.sin(phi)

    R2 = x*(R*np.sin(phi) + y*np.cos(phi))
    X = R2 * np.sin(THETA)
    Y = R2*np.cos(THETA)
    Z = x*(R*np.cos(phi) - y*np.sin(phi))


    # e = SREvaluator(x_train=x_train, y_train=y_train, loss='mse')
    
    
    
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Reds')
    
    
    
    plt.show()

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
        print('Starting for roses')
        base('test')
        roses('test')
    if len(sys.argv)==2:
        print('Starting evolution from random genome')
        evolve(sys.argv[1])
    elif len(sys.argv)==3:
        print('Starting evolution from genome saved in ', sys.argv[2])
        evo(sys.argv[1], genome=sys.argv[2])