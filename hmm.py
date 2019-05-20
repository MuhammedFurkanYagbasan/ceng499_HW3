##########################################################
#
# Muhammed Furkan YAĞBASAN - 2099505
# 
# 19.05.2019 - ceng499 HW3
# 
##########################################################

import numpy as np

def forward(A, B, pi, O):
	"""
	Parameters:
		A: state transition probabilities(NxN)
		B: observation probabilites(NxM)
		pi: initial state probabilities(N)
		O: sequence of observations(T) where
		   observations are just indices for the columns of B
		where N is the number of states,
			  M is the number of possible observations, and
			  T is the sequence length.
	Return:
		Probability of the observation sequence given the model(A, B, pi)
	"""
	
	########## Write your code here ##########
	# implement forward algorithm to find
	# the probability of the given observation
	# sequence given the model.
	##########################################

	N = A.shape[0]
	M = B.shape[1]
	T = O.shape[0]

	trellisDiagram = np.zeros((T,N))

	
	for i in range(T):
		for j in range(N):
			if(i == 0):
				trellisDiagram[i][j] = pi[j] * B[j][O[0]]
			else:
				for preS in range(N):
					trellisDiagram[i][j] += trellisDiagram[i-1][preS] * A[preS][j]
				trellisDiagram[i][j] *= B[j][O[i]]

	result = 0
	for i in range(N):
		result += trellisDiagram[T-1][i]

	return result


def viterbi(A, B, pi, O):
	"""
	Parameters:
		A: state transition probabilities(NxN)
		B: observation probabilites(NxM)
		pi: initial state probabilities(N)
		O: sequence of observations(T) where
		   observations are just indices for the columns of B
		where N is the number of states,
			  M is the number of possible observations, and
			  T is the sequence length.
	Return:
		The most likely state sequence given model(A, B, pi) and
		observation sequence. It should be a numpy array
		with size T. It includes state indices according to
		A's indices. For example: [1,2,1,1,0,4]
	"""
	
	########## Write your code here ##########
	# implement viterbi algorithm to find
	# the most likely state sequence of length
	# T given model and observation sequence.
	##########################################	

	N = A.shape[0]
	M = B.shape[1]
	T = O.shape[0]

	trellisDiagram = np.zeros((T,N))
	traceBack = np.zeros((T,N))

	for i in range(T):
		for j in range(N):
			if(i == 0):
				trellisDiagram[i][j] = pi[j] * B[j][O[0]]
				traceBack[i][j] = 0
			else:
				for preS in range(N):
					current = trellisDiagram[i-1][preS] * A[preS][j]
					if(current>trellisDiagram[i][j]):
						trellisDiagram[i][j] = current
						traceBack[i][j] = preS
				trellisDiagram[i][j] *= B[j][O[i]]
			
	result = np.zeros((T,))

	lastState = int(np.argmax(trellisDiagram[T-1]))
	result[T-1] = lastState

	for t in range(T-1):
		lastState = int(traceBack[T-1-t][lastState])
		result[T-1-t-1] = lastState

	return result
