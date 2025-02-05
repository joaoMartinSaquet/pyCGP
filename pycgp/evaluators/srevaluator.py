from ..cgp import CGP
from .evaluator import Evaluator

import numpy as np

class SREvaluator(Evaluator):

    def __init__(self, x_train, y_train, loss = 'mse'):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train

        if loss == 'mse':
            self.loss = lambda y_cgp, y_train: np.mean((y_cgp - y_train)**2)
        elif loss == 'mae':
            self.loss = lambda y_cgp, y_train: np.mean(np.abs(y_cgp - y_train))
    

    def evaluate(self, cgp : CGP, it):

        y_cgp = cgp.run(self.x_train)
        loss = self.loss(y_cgp, self.y_train)

        # -loss because we want to minimize the error
        return -loss
    
    
