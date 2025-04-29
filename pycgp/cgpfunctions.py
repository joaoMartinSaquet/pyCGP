import numpy as np
import math

def f_sum(args, const_params):
	return (args[0] + args[1])

def f_aminus(args, const_params):
	return (abs(args[0] - args[1]))

def f_mult(args, const_params):
	return args[0] * args[1]

def f_exp(args, const_params):
	return np.exp(args[0])

def f_abs(args, const_params):
	return abs(args[0])

def f_sqrt(args, const_params):
	return np.sqrt(abs(args[0]))

def f_sqrtxy(args, const_params):
	return np.sqrt(args[0] * args[0] + args[1] * args[1]) / np.sqrt(2.0)

def f_squared(args, const_params):
	return args[0] * args[0]

def f_pow(args, const_params):
	return pow(abs(args[0]), abs(args[1]))

def f_one(args, const_params):
	return 1.0

def f_const(args, const_params):
	return const_params[0]

def f_zero(args, const_params):
	return 0.0

def f_inv(args, const_params):

	np.divide(args[0],abs(args[0]),out=np.zeros_like(args[0]),where=args[0]!=0)
	
def f_gt(args, const_params):
	return  (args[0] > args[1]).astype(np.float128) * args[0] # return args[0] if he is greater than args[1] else 0

def f_acos(args, const_params):
	return np.acos(args[0]) / np.pi


def f_asin(args, const_params):
	return 2.0 * np.asin(args[0])/ np.pi

def f_atan(args, const_params):
	return 2.0 * np.atan(args[0]) / np.pi

def f_min(args, const_params):
	return np.min(args)

def f_max(args, const_params):
	return np.max(args)

def f_round(args, const_params):
	return np.round(args[0])

def f_floor(args, const_params):
	return np.floor(args[0])

def f_ceil(args, const_params):
	return np.ceil(args[0])

def f_sin(args, const_params):
	return np.sin(args[0])

def f_cos(args, const_params):
	return np.cos(args[0])

def f_mod(args, const_params):
	return np.mod(args[0], args[1])

def f_div(args, cons_params):
	if np.any(args[1] == 0):
		return args[0]
	else:
		return np.divide(args[0], args[1])
	
def f_log(args, const_params):
	return np.log1p(abs(args[0])).astype(np.uint8)

def f_tanh(args, const_params):
	return np.tanh(args[0])

def f_lt(args, const_params):
	return  (args[0] < args[1]).astype(np.float128)

def f_atan2(args, const_params):
	return np.arctan2(args[0], args[1])