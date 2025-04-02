import os
import numpy as np
from .cgp import CGP
import cma


# from .evaluators.evaluator import Evaluator
from joblib import Parallel, delayed


CMA_OPTIONS = {
    'popsize': 50,  # Population size
    'tolfun': 1e-3,  # Stop if the function value difference is less than this
    'maxiter': 100,  # Maximum number of iterations
	'verb_log' : 0,
	'verb_disp' : 0
}

def selection_wheel(pop, pop_fit, n):


    # 1/x => -1000  is then close to 0
    prob_fit = 1/(abs(pop_fit) + 0.1) 
    prob_fit = np.cumsum(prob_fit/np.sum(prob_fit))

    new_pop = []
    new_fit = []
    for i in range(n):
        p = np.random.random()
        ind = len(np.where(prob_fit<p)[0])
        new_pop.append(pop[ind])
        new_fit.append(pop_fit[ind])

    return new_pop, new_fit

def selection_elitism(pop, pop_fit, n):


    ind_hof = np.argsort(pop_fit)[::-1]

    return pop[ind_hof[:n]], pop_fit[ind_hof[:n]]
	

def selection_tournament(pop, pop_fit, n, tournament_size = 2):


	new_pop = []
	new_fit = []
		
	for i in range(n):
		tournament_ids = np.random.randint(0, len(pop), tournament_size)

		tournament_fit = pop_fit[tournament_ids]

		winner_id = tournament_ids[np.argmax(tournament_fit)]

		new_pop.append(pop[winner_id])
		new_fit.append(pop_fit[winner_id])

	return new_pop, new_fit


class CGPES_1l:
	# TODO add validation parameter
	def __init__(self, num_offsprings, mutation_rate_nodes, mutation_rate_outputs, father, evaluator, folder='genomes', num_cpus = 1):
		self.num_offsprings = num_offsprings
		self.mutation_rate_nodes = mutation_rate_nodes
		self.mutation_rate_outputs = mutation_rate_outputs
		self.father = father
		#self.num_mutations = int(len(self.father.genome) * self.mutation_rate)
		self.evaluator = evaluator
		self.num_cpus = num_cpus
		self.folder = folder
		if self.num_cpus > 1:
			self.evaluator_pool = []
			for i in range(self.num_offsprings):
				self.evaluator_pool.append(self.evaluator.clone())
		self.initialized = False
		self.fitness_history = []

	def initialize(self):
		if not os.path.isdir(self.folder):
			os.mkdir(self.folder)
		self.logfile = open(self.folder + '/out.txt', 'w')
		self.current_fitness = self.evaluator.evaluate(self.father, 0)
		self.father.save(self.folder + '/cgp_genome_0_' + str(self.current_fitness) + '.txt')
		self.offsprings = np.empty(self.num_offsprings, dtype=type(self.father))
		self.offspring_fitnesses = np.zeros(self.num_offsprings, dtype=float)
		self.initialized = True
		self.it = 0

	def run(self, num_iteration):
		if not self.initialized:
			self.initialize()

		for it in range(num_iteration):
			self.it += 1
			#generate offsprings
			if self.num_cpus == 1:
				for i in range(0, self.num_offsprings):
					self.offsprings[i] = self.father.clone()
					#self.offsprings[i].mutate(self.num_mutations)
					self.offsprings[i].mutate_per_gene(self.mutation_rate_nodes, self.mutation_rate_outputs)
#					self.offsprings[i].goldman_mutate_2()
					self.offspring_fitnesses[i] = self.evaluator.evaluate(self.offsprings[i], self.it)
			else:
				for i in range(self.num_offsprings):
					self.offsprings[i] = self.father.clone()
					#self.offsprings[i].mutate(self.num_mutations)
					#self.offsprings[i].mutate_per_gene(self.mutation_rate_nodes, self.mutation_rate_outputs)
					self.offsprings[i].goldman_mutate()
				def offspring_eval_task(offspring_id):
					return self.evaluator_pool[offspring_id].evaluate(self.offsprings[offspring_id], self.it)
				self.offspring_fitnesses = Parallel(n_jobs = self.num_cpus)(delayed(offspring_eval_task)(i) for i in range(self.num_offsprings)) 
			#get the best fitness
			best_offspring = np.argmax(self.offspring_fitnesses)
			if not self.evaluator.is_cacheable(self.it):
				self.current_fitness = self.evaluator.evaluate(self.father, self.it)
			#compare to father
			self.father_was_updated = False
			if self.offspring_fitnesses[best_offspring] >= self.current_fitness:
				self.current_fitness = self.offspring_fitnesses[best_offspring]
				self.father = self.offsprings[best_offspring]
				self.father_was_updated = True
			# display stats
			print(self.it, '\t', self.current_fitness, '\t', self.father_was_updated, '\t', self.offspring_fitnesses)
			self.logfile.write(str(self.it) + '\t' + str(self.current_fitness) + '\t' + str(self.father_was_updated) + '\t' + str(self.offspring_fitnesses) + '\n')
			self.logfile.flush()
			self.fitness_history.append(self.current_fitness)
			print('====================================================')
			if self.father_was_updated:
				#print(self.father.genome)
				self.father.save(self.folder + '/cgp_genome_' + str(self.it) + '_' + str(self.current_fitness) + '.txt')


class CGPES_ml:
	# TODO add validation parameter
	def __init__(self, num_father, num_offsprings, mutation_rate_nodes, mutation_rate_outputs, 
			  mutation_rate_const_params, father, evaluator, folder='genomes', num_cpus = 1):
		self.lbda = num_offsprings
		self.mutation_rate_nodes = mutation_rate_nodes
		self.mutation_rate_outputs = mutation_rate_outputs
		self.mutation_rate_const_params = mutation_rate_const_params

		self.mu = num_father
		self.hof = father
		#self.num_mutations = int(len(self.father.genome) * self.mutation_rate)
		self.evaluator = evaluator
		self.num_cpus = num_cpus
		self.folder = folder
		if self.num_cpus > 1:
			self.evaluator_pool = []
			for i in range(self.lbda):
				self.evaluator_pool.append(self.evaluator.clone())
		self.initialized = False
		self.fitness_history = []

	def initialize(self):
		self.it = 0
		if not os.path.isdir(self.folder):
			os.mkdir(self.folder)
		self.logfile = open(self.folder + '/out.txt', 'w')
		self.hof_fit = []
		for hof in self.hof:
			self.hof_fit.append(self.evaluator.evaluate(hof, 0))
		self.hof[np.argmax(self.hof_fit)].save(self.folder + '/cgp_genome_0_' + str(self.it) + '.txt')
		self.offsprings = np.empty(self.lbda, dtype=type(self.hof[0]))
		self.offspring_fitnesses = np.zeros(self.lbda, dtype=float)
		self.initialized = True


	def run(self, num_iteration, term_criteria):
		if not self.initialized:
			self.initialize()

		for it in range(num_iteration):
			self.it += 1
			#generate offsprings
			if self.num_cpus == 1:

				for i in range(0, self.lbda):
					
					# select the parents to mutate from randomly (a parent may not reproduce)
					j = np.random.randint(0, self.mu)
					self.offsprings[i] = self.hof[j].clone()
					#self.offsprings[i].mutate(self.num_mutations)
					self.offsprings[i].mutate_per_gene(self.mutation_rate_nodes, self.mutation_rate_outputs, self.mutation_rate_const_params)
					# self.offsprings[i].goldman_mutate()
#					self.offsprings[i].goldman_mutate_2()


					if self.offsprings[i].cst_to_optimize:
						self.offsprings[i].cst_to_optimize = False
						# cmaes optimization

						#CMA(mean, sigma, [self.offsprings[i].const_min, self.offsprings[i].const_max])
						opt = cma.CMAEvolutionStrategy(self.offsprings[i].cst_table, 1.0, CMA_OPTIONS)						# self.offspring_fitnesses[i] = self.evaluator.evaluate(self.offsprings[i], self.it)
						def obj(x):
							self.offsprings[i].cst_table = x
							# self.offsprings[i].cst_to_optimize = True
							return 	-self.evaluator.evaluate(self.offsprings[i], self.it)	# evaluate is higher is better and cma try to get min 
						
						opt.optimize(obj)
						self.offsprings[i].cst_to_optimize = False
						self.offsprings[i].cst_table = opt.best.x
						self.offspring_fitnesses[i] = -opt.best.f
					else:
						self.offspring_fitnesses[i] = self.evaluator.evaluate(self.offsprings[i], self.it)
			else:
				# deprecated for the moment
				for i in range(self.lbda):
					self.offsprings[i] = self.hof.clone()
					#self.offsprings[i].mutate(self.num_mutations)
					#self.offsprings[i].mutate_per_gene(self.mutation_rate_nodes, self.mutation_rate_outputs)
					self.offsprings[i].goldman_mutate()
				def offspring_eval_task(offspring_id):
					return self.evaluator_pool[offspring_id].evaluate(self.offsprings[offspring_id], self.it)
				self.offspring_fitnesses = Parallel(n_jobs = self.num_cpus)(delayed(offspring_eval_task)(i) for i in range(self.lbda)) 
			#get the best fitness
			# best_offspring = np.argmax(self.offspring_fitnesses)
			if not self.evaluator.is_cacheable(self.it):
				self.hof_fit = self.evaluator.evaluate(self.hof, self.it)

			pop  = np.hstack((self.hof, self.offsprings))
			pop_fit = np.hstack((self.hof_fit, self.offspring_fitnesses))

			# selction piped wheel for now 
			# self.hof, self.hof_fit = selection_wheel(pop, pop_fit, self.mu)

			# eltism selection
			# self.hof, self.hof_fit = selection_elitism(pop, pop_fit, self.mu)

			# tournament selection
			self.hof, self.hof_fit = selection_tournament(pop, pop_fit, self.mu, tournament_size = int((self.mu + self.lbda)/4)) 


			# display stats
			if self.it % 10 == 0:
				print(self.it, '\t mean hof fit ', np.mean(self.hof_fit), '\t best hof fit ', str(np.max(self.hof_fit)), '\t', self.offspring_fitnesses)
				self.logfile.write(str(self.it) + '\t' + str(self.hof_fit) + '\t' + str(self.hof_fit) + '\n')
				self.logfile.flush()
			self.fitness_history.append(np.max(self.hof_fit))

			if np.min(np.abs(np.max(self.hof_fit))) <= term_criteria:
				break
			# find how to save now 
			# print('====================================================')
			# if self.father_was_updated:
			# 	#print(self.father.genome)
			# 	self.hof.save(self.folder + '/cgp_genome_' + str(self.it) + '_' + str(self.hof_fit) + '.txt')



# selection algorithms

