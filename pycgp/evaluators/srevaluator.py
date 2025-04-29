from ..cgp import CGP, CGPFunc, CGP_with_cste
from ..cgpfunctions import *
from ..cgpes import *
from .evaluator import Evaluator

import numpy as np
import sympy as sp

DEFAULT_SR_LIBRARY = [
            CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGPFunc(f_sin, 'sin', 1, 0, 'sin'),
            CGPFunc(f_cos, 'cos', 1, 0, 'cos'),
            CGPFunc(f_div, 'div', 2, 0, '/'),
            # CGPFunc(f_const, 'c', 0, 1, 'c')
            ]


class SREvaluator(Evaluator):

    def __init__(self, x_train, y_train, n_inputs, n_outputs, col = 30, row=1, library = DEFAULT_SR_LIBRARY, loss = 'mse'):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.col = col
        self.row = row
        self.library = library
        if loss == 'mse':
            self.loss = lambda y_cgp, y_train: np.mean((y_cgp - y_train)**2)
        elif loss == 'mae':
            self.loss = lambda y_cgp, y_train: np.mean(np.abs(y_cgp - y_train))
        elif loss == 'rmse':
            self.loss = lambda y_cgp, y_train: np.sqrt(np.mean((y_cgp - y_train)**2))    
        elif loss == 'norm':
            self.loss = lambda y_cgp, y_train: np.linalg.norm(y_cgp - y_train)/y_cgp.shape[0]
        self.input_shape = (max(x_train.shape), )


    def evaluate(self, cgp : CGP_with_cste, it):

        y_cgp = cgp.run(self.x_train).T # numpy reshape suck so hard 

        loss = self.loss(y_cgp, self.y_train)

        if not np.isfinite(loss):
            loss = 1000000

        # -loss because we want to minimize the error
        return -loss

    def clone(self):
        return SREvaluator(self.x_train, self.y_train)
    
    def evolve(self, mu, nb_ind = 4, num_csts = 0, mutation_rate_nodes = 0.2, mutation_rate_outputs=0.2, mutation_rate_const_params=0.01, n_cpus=1, n_it=500, folder_name='test', term_criteria=1e-4, random_genomes=True):

        if mu > 1: 
            if num_csts > 0:
                self.hof = [CGP_with_cste.random(self.n_inputs, self.n_outputs, num_csts, self.col, self.row, self.library, 
                                    self.col, False, const_min=-10, const_max=10, input_shape=self.input_shape, dtype='float')
                                        for i in range(mu)]
                
                if not random_genomes:
                    # get the prior knowledge of the problem (fitts dx = y0 and dy = y1)
                    for ind in self.hof:
                        ind.genome[-2] = 0 # y0 = dx
                        ind.genome[-1] = 1 # y1 = dy
                es = CGPES_ml(mu, nb_ind, mutation_rate_nodes, mutation_rate_outputs, mutation_rate_const_params, self.hof, self, folder_name, n_cpus)
                
            else:
                self.hof = [CGP_with_cste.random(self.n_inputs, self.n_outputs, num_csts, self.col, self.row, self.library, 
                                   self.col, False, const_min=-10, const_max=10, input_shape=self.input_shape, dtype='float')
                                    for i in range(mu)]
                es = CGPES_ml(mu, nb_ind, mutation_rate_nodes, mutation_rate_outputs, mutation_rate_const_params, self.hof, self, folder_name, n_cpus)
            es.run(n_it, term_criteria=term_criteria)
            fit_history = es.fitness_history
            best = es.hof[np.argmax(es.hof_fit)]
        else :
            self.hof = CGP_with_cste.random(self.n_inputs, self.n_outputs, self.col, self.row, self.library, 
                                   self.col, False, const_min=0, const_max=1, input_shape=self.input_shape, dtype='float')
            es = CGPES_1l(nb_ind, mutation_rate_nodes, mutation_rate_outputs, self.cgpFather, self, folder_name, n_cpus)
            es.run(n_it)
            fit_history = es.fitness_history
            best = es.father
        
        self.best = best
        
        # print("best fitness : ", fit_history[-1])
        return best, fit_history
        
    def best_logs(self, input_names, output_names):
        
        
        out, infix_out = self.best.to_function_string(input_names, output_names)
    
        print("best raw equations : ",out)
        out_equation = []
        for o in infix_out:
            print("HOF best simplified : ", sp.simplify(o))
            out_equation.append(sp.simplify(o))
        return out_equation

    def best_evaluate(self, x):
        return self.best.run(x)
