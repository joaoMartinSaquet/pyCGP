from ..cgp import CGP
from ..cgpfunctions import *
from ..cgpes import *
from .evaluator import Evaluator

import numpy as np
import sympy as sp

DEFAULT_SR_LIBRARY = [CGP.CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGP.CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGP.CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGP.CGPFunc(f_sin, 'sin', 1, 0, 'sin'),
            CGP.CGPFunc(f_cos, 'cos', 1, 0, 'cos'),
            ]


class SREvaluator(Evaluator):

    def __init__(self, x_train, y_train, n_inputs, n_outputs, col = 30, row=1, library = DEFAULT_SR_LIBRARY, loss = 'mse', mu=4, es="m+l"):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train

        if loss == 'mse':
            self.loss = lambda y_cgp, y_train: np.mean((y_cgp - y_train)**2)
        elif loss == 'mae':
            self.loss = lambda y_cgp, y_train: np.mean(np.abs(y_cgp - y_train))

        input_shape = (max(x_train.shape), )

        self.mu = None
        self.es = es
        if es == "m+l":
            self.mu = mu
            self.hof = [CGP.random(n_inputs, n_outputs,  col, row, library, col, False, const_min=0, const_max=1, input_shape=input_shape, dtype='float')
                                    for i in range(mu)]

        else:
            self.hof = CGP.random(n_inputs, n_outputs,  col, row, library, col, False, const_min=0, const_max=1, input_shape=input_shape, dtype='float')


    def evaluate(self, cgp : CGP, it):

        y_cgp = cgp.run(self.x_train).reshape(self.y_train.shape)
        loss = self.loss(y_cgp, self.y_train)

        if not np.isfinite(loss):
            loss = 1000

        # -loss because we want to minimize the error
        return -loss

    def clone(self):
        return SREvaluator(self.x_train, self.y_train)
    
    def evolve(self, mu, nb_ind = 4, mutation_rate_nodes = 0.2, mutation_rate_outputs=0.2, n_cpus=1, n_it=500, folder_name='test', term_criteria=1e-6):

        self.mu = mu
        if self.es == "m+l":
            es = CGPES_ml(self.mu, nb_ind, mutation_rate_nodes, mutation_rate_outputs, self.hof, self, folder_name, n_cpus)
            es.run(n_it, term_criteria=term_criteria)
            fit_history = es.fitness_history
            best = es.hof[np.argmax(es.hof_fit)]
        else :
            es = CGPES_1l(nb_ind, mutation_rate_nodes, mutation_rate_outputs, self.cgpFather, self, folder_name, n_cpus)
            es.run(n_it)
            fit_history = es.fitness_history
            best = es.father
        
        self.best = best
            
        return best, fit_history
        
    def best_logs(self, input_names, output_names):
        
        
        out, infix_out = self.best.to_function_string(input_names, output_names)
    
        print("best raw equations : ",out)
        for o in infix_out:
            print("HOF best simplified : ", sp.simplify(o))


    def best_evaluate(self, x):
        return self.best.run(x)
