from ..cgp import CGP
from ..cgpfunctions import *
from ..cgpes import CGPES
from .evaluator import Evaluator

import numpy as np


DEFAULT_SR_LIBRARY = [CGP.CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGP.CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGP.CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGP.CGPFunc(f_sin, 'sin', 1, 0, 'sin'),
            CGP.CGPFunc(f_cos, 'cos', 1, 0, 'cos'),
            ]


class SREvaluator(Evaluator):

    def __init__(self, x_train, y_train, n_inputs, n_outputs, col = 30, row=1, library = DEFAULT_SR_LIBRARY, loss = 'mse'):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train

        if loss == 'mse':
            self.loss = lambda y_cgp, y_train: np.mean((y_cgp - y_train)**2)
        elif loss == 'mae':
            self.loss = lambda y_cgp, y_train: np.mean(np.abs(y_cgp - y_train))
    
        self.cgpFather = CGP.random(n_inputs, n_outputs,  col, row, library, col, False, const_min=0, const_max=1, input_shape=x_train.shape, dtype='float')


    def evaluate(self, cgp : CGP, it):

        y_cgp = cgp.run(self.x_train).reshape(self.y_train.shape)
        loss = self.loss(y_cgp, self.y_train)

        # -loss because we want to minimize the error
        return -loss

    def clone(self):
        return SREvaluator(self.x_train, self.y_train)
    
    def evolve(self, nb_ind = 4, mutation_rate_nodes = 0.1, mutation_rate_outputs=0.2, n_cpus=1, n_it=500, folder_name='test'):

        es = CGPES(nb_ind, mutation_rate_nodes, mutation_rate_outputs, self.cgpFather, self, folder_name, n_cpus)
        es.run(n_it)

        return es.father
    
    
