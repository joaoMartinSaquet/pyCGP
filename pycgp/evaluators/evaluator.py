from ..cgp import CGP

class Evaluator:
	def evaluate(self, cgp, it):
		raise NotImplementedError('evaluation method not implemented')

	def is_cacheable(self, it):
		return True

	def clone(self):
		raise NotImplementedError('clone method not implemented')

