# Function fitting using gradient descent.
# This works by first getting random inputs, and then minmizing the amount of loss by moving the m and b variables in direction of least loss.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import random

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

class OptimizerOutput:
	def __init__(self, answers, loss, meta, inputs, func):
		self.answers = answers
		self.loss = loss
		self.history = meta[0]
		self.iterations = meta[1]
		self.inputs = inputs
		self.func = func
	def plotLossMovement(self):
		ax.plot(self.history[0], self.history[1], self.history[2])
		plt.show()
	def plotFitting(self):
		for i in range(len(self.history[0])):
			if i % 3 == 0:
				vals = self.func(self.history[0][i], self.history[1][i], self.inputs[0])
				plt.plot(self.inputs[0], vals)
		plt.scatter(self.inputs[0], self.inputs[1])
		plt.show()
	def plotLossChart(self):
		plt.plot([i for i in range(len(self.history[2]))], self.history[2])
		plt.show()

class MinimizerOutput:
	def __init__(self, answers, loss, meta, func, constraints, finished):
		self.answers = answers
		self.loss = loss
		self.history = meta[0]
		self.iterations = meta[1]
		self.func = func
		self.constraints = constraints
		self.finished = finished
	def plotLossChart(self):
		plt.plot([i for i in range(len(self.history[2]))], self.history[2])
		plt.show()

def frange(_min, _max, step=1):
	output = []
	current = _min
	while current < _max:
		output.append(current)
		current += step
	return np.array(output)

def stretch(X, y, amount):
	return X/amount, y/amount

def linear(x, y, X):
	return x * X + y

def quadratic(x, y, X):
	return x * X ** 2 + y * X

def loss(xval, yval, X, y, func):
	return np.sum(np.square(func(xval, yval, X) - y)) / len(X) 

def visualizeLoss(X, y, func, lossFunc=loss, xmin=-10, ymin=-10, xmax=10, ymax=10, dx=1, dy=1, vis=True):
	xs = np.ravel([[i] * len(frange(xmin, xmax, dx)) for i in frange(ymin, ymax, dy)])
	ys = np.array(list(frange(ymin, ymax, dy)) * len(frange(xmin, xmax, dx)))
	zs = []
	for xp, yp in zip(xs, ys):
		zs.append(lossFunc(xp, yp, X, y, func))
	if vis:
		ax.scatter(xs, ys, zs)
		plt.show()
	else:
		return xs, ys, zs

def visualizeFunction(func, xmin=-10, ymin=-10, xmax=10, ymax=10, dx=1, dy=1, vis=True):
	xs = np.ravel([[i] * len(frange(xmin, xmax, dx)) for i in frange(ymin, ymax, dy)])
	ys = np.array(list(frange(ymin, ymax, dy)) * len(frange(xmin, xmax, dx)))
	zs = []
	for xp, yp in zip(xs, ys):
		zs.append(func(xp, yp))
	if vis:
		ax.scatter(xs, ys, zs)
		plt.show()
	else:
		return xs, ys, zs

def visualize2D(func, xmin=-10, xmax=10, dx=1, vis=True):
	xs = frange(xmin, xmax, dx)
	ys = [func(i) for i in xs]
	if vis:
		plt.plot(xs, ys)
		plt.show()
	else:
		return xs, ys

def _findMinimalLossDirection(lossFunc, currentX, currentY, dx, dy, currentLoss, func, X, y):
	# Check all 8 possibilities by taking steps in dx and dy amounts
	lossChanges = (lossFunc(currentX + dx, currentY + dy, X, y, func) - currentLoss, # +dx, +dy
	lossFunc(currentX + dx, currentY - dy, X, y, func) - currentLoss, # +dx, -dy
	lossFunc(currentX + dx, currentY, X, y, func) - currentLoss, # +dx, 0
	lossFunc(currentX - dx, currentY + dy, X, y, func) - currentLoss, # -dx, +dy
	lossFunc(currentX - dx, currentY - dy, X, y, func) - currentLoss, # -dx, -dy
	lossFunc(currentX - dx, currentY, X, y, func) - currentLoss, # -dx, 0
	lossFunc(currentX, currentY + dy, X, y, func) - currentLoss, # 0, +dy
	lossFunc(currentX, currentY - dy, X, y, func) - currentLoss) # 0, -dy
	# Find minimum loss value
	minLoss = min(lossChanges)
	# Return direction for x and y
	possibleDirs = ((dx, dy), (dx, -dy), (dx, 0), (-dx, dy), (-dx, -dy), (-dx, 0), (0, dy), (0, -dy))
	directionToGo = possibleDirs[lossChanges.index(minLoss)]
	return directionToGo

def fit(lossFunc, X, y, func=linear, startx=np.random.rand(1)[0], starty=np.random.rand(1)[0], dx=0.01, dy=0.01, maxIter=3000, lossThresh=0.01, calculateMovement=True, verbose=False):
	# Starting values
	iteration = 0
	currentX = startx
	currentY = starty
	currentLoss = lossFunc(currentX, currentY, X, y, func)
	historyX = []
	historyY = []
	historyZ = []

	while iteration < maxIter:
		if verbose:
			print('Iteration:', iteration, 'X:', currentX, 'Y:', currentY, 'Loss:', currentLoss)
		if iteration % 20 == 0:
			historyX.append(currentX)
			historyY.append(currentY)
			historyZ.append(currentLoss)
		# Take step in x and y direction, and calculate loss increase/decrease (not slope because we don't care about magnitude), then find direction to move in
		if calculateMovement:
			movementAmountX = np.sqrt(abs(currentLoss)) * dx / 2
			movementAmountY = np.sqrt(abs(currentLoss)) * dy / 2
			moveDir = _findMinimalLossDirection(lossFunc, currentX, currentY, movementAmountX, movementAmountY, currentLoss, func, X, y)
		else:
			moveDir = _findMinimalLossDirection(lossFunc, currentX, currentY, dx, dy, currentLoss, func, X, y)
		# Move in that direction
		newX = currentX + moveDir[0]
		newY = currentY + moveDir[1]
		currentX = newX
		currentY = newY
		# Recalculate loss
		newLoss = lossFunc(currentX, currentY, X, y, func)
		currentLoss = newLoss
		if currentLoss <= lossThresh:
			break
		iteration += 1

	return OptimizerOutput((currentX, currentY), currentLoss, ((historyX, historyY, historyZ), iteration), (X, y), func)

def findAnswer(output, X, func=linear):
	return func(output.answers[0], output.answers[1], X)

def _findMinimizeDirection(x, y, function, currentLoss, dx, dy, restriction, restrictedMovements=()):
	# Check all 8 possibilities
	xyChanges = [(dx, dy), (dx, -dy), (dx, 0), (-dx, dy), (-dx, -dy), (-dx, 0), (0, dy), (0, -dy)]
	for restrictedMovement in restrictedMovements:
		xyChanges.remove(restrictedMovement)
	if len(xyChanges) == 0:
		return None
	# Calculate x and y values
	lossChanges = []
	for xyChange in xyChanges:
		lossChanges.append(function(x + xyChange[0], y + xyChange[1]))
	# Find minimum loss value
	minLoss = min(lossChanges)
	# Return direction for x and y
	directionToGo = xyChanges[lossChanges.index(minLoss)]
	# Make sure constraints work
	if restriction != None:
		if restriction(x+directionToGo[0], y+directionToGo[1], minLoss) == False:
			return _findMinimizeDirection(x, y, function, currentLoss, dx, dy, restriction, tuple(list(restrictedMovements) + [directionToGo]))

	return directionToGo

def minimize(function, goal, restriction=None, startx=np.random.rand(1)[0], starty=np.random.rand(1)[0], dx=0.01, dy=0.01, maxIter=3000, calculateMovement=True, verbose=False):
	# Starting values
	iteration = 0
	currentX = startx
	currentY = starty
	currentLoss = function(currentX, currentY)
	historyX = []
	historyY = []
	historyZ = []

	while iteration < maxIter:
		if verbose:
			print('Iteration:', iteration, 'X:', currentX, 'Y:', currentY, 'Loss:', currentLoss)
		if iteration % 20 == 0:
			historyX.append(currentX)
			historyY.append(currentY)
			historyZ.append(currentLoss)
		# Take step in x and y direction, and calculate loss increase/decrease (not slope because we don't care about magnitude), then find direction to move in
		if calculateMovement:
			movementAmountX = np.sqrt(abs(currentLoss)) * dx / 2
			movementAmountY = np.sqrt(abs(currentLoss)) * dy / 2
			moveDir = _findMinimizeDirection(currentX, currentY, function, currentLoss, movementAmountX, movementAmountY, restriction)
		else:
			moveDir = _findMinimizeDirection(currentX, currentY, function, currentLoss, dx, dx, restriction)

		if moveDir == None:
			return MinimizerOutput((currentX, currentY), currentLoss, ((historyX, historyY, historyZ), iteration), function, restriction, False)
		# Move in that direction
		newX = currentX + moveDir[0]
		newY = currentY + moveDir[1]
		currentX = newX
		currentY = newY
		# Recalculate loss
		newLoss = function(currentX, currentY)
		currentLoss = newLoss
		if currentLoss <= goal:
			break
		iteration += 1

	return MinimizerOutput((currentX, currentY), currentLoss, ((historyX, historyY, historyZ), iteration), function, restriction, True)

def _findMaximizeDirection(x, y, function, currentLoss, dx, dy, restriction, restrictedMovements=()):
	# Check all 8 possibilities
	xyChanges = [(dx, dy), (dx, -dy), (dx, 0), (-dx, dy), (-dx, -dy), (-dx, 0), (0, dy), (0, -dy)]
	for restrictedMovement in restrictedMovements:
		xyChanges.remove(restrictedMovement)
	if len(xyChanges) == 0:
		return None
	# Calculate x and y values
	lossChanges = []
	for xyChange in xyChanges:
		lossChanges.append(function(x + xyChange[0], y + xyChange[1]))
	# Find minimum loss value
	minLoss = max(lossChanges)
	# Return direction for x and y
	directionToGo = xyChanges[lossChanges.index(minLoss)]
	# Make sure constraints work
	if restriction != None:
		if restriction(x+directionToGo[0], y+directionToGo[1], minLoss) == False:
			return _findMaximizeDirection(x, y, function, currentLoss, dx, dy, restriction, tuple(list(restrictedMovements) + [directionToGo]))

	return directionToGo

def maximize(function, goal, restriction=None, startx=np.random.rand(1)[0], starty=np.random.rand(1)[0], dx=0.01, dy=0.01, maxIter=3000, calculateMovement=True, verbose=False):
	# Starting values
	iteration = 0
	currentX = startx
	currentY = starty
	currentLoss = function(currentX, currentY)
	historyX = []
	historyY = []
	historyZ = []

	while iteration < maxIter:
		if verbose:
			print('Iteration:', iteration, 'X:', currentX, 'Y:', currentY, 'Loss:', currentLoss)
		if iteration % 20 == 0:
			historyX.append(currentX)
			historyY.append(currentY)
			historyZ.append(currentLoss)
		# Take step in x and y direction, and calculate loss increase/decrease (not slope because we don't care about magnitude), then find direction to move in
		if calculateMovement:
			movementAmountX = np.sqrt(abs(currentLoss)) * dx / 2
			movementAmountY = np.sqrt(abs(currentLoss)) * dy / 2
			moveDir = _findMaximizeDirection(currentX, currentY, function, currentLoss, movementAmountX, movementAmountY, restriction)
		else:
			moveDir = _findMaximizeDirection(currentX, currentY, function, currentLoss, dx, dy, restriction)

		if moveDir == None:
			return MinimizerOutput((currentX, currentY), currentLoss, ((historyX, historyY, historyZ), iteration), function, restriction, False)
		# Move in that direction
		newX = currentX + moveDir[0]
		newY = currentY + moveDir[1]
		currentX = newX
		currentY = newY
		# Recalculate loss
		newLoss = function(currentX, currentY)
		currentLoss = newLoss
		if currentLoss >= goal:
			break
		iteration += 1

	return MinimizerOutput((currentX, currentY), currentLoss, ((historyX, historyY, historyZ), iteration), function, restriction, True)

def _findMinimizeDirection2D(x, function, currentLoss, dx):
	xChanges = [x + dx, x - dx]
	lossChanges = [function(i) for i in xChanges]
	directionToGo = xChanges[lossChanges.index(min(lossChanges))]
	return directionToGo

def minimize2D(func, goal, startx=np.random.rand(1)[0], dx=0.01, maxIter=3000, calculateMovement=True, verbose=False):
	iteration = 0
	currentX = startx
	currentLoss = func(currentX)
	historyX = []
	historyY = []

	while iteration < maxIter:
		if verbose:
			print('Iteration:', iteration, 'X:', currentX, 'Loss:', currentLoss)
		if iteration % 20 == 0:
			historyX.append(currentX)
			historyY.append(currentLoss)
		# Take step in x direction, and calculate loss increase/decrease (not slope because we don't care about magnitude), then find direction to move in
		if calculateMovement:
			movementAmountX = np.sqrt(abs(currentLoss)) * dx / 2
			moveDir = _findMinimizeDirection2D(currentX, func, currentLoss, movementAmountX)
		else:
			moveDir = _findMinimizeDirection2D(currentX, func, currentLoss, dx)
		# Move in that direction
		newX = moveDir
		currentX = newX
		# Recalculate loss
		newLoss = func(currentX)
		currentLoss = newLoss
		if currentLoss <= goal:
			break
		iteration += 1

	return MinimizerOutput((currentX), currentLoss, ((historyX, historyY), iteration), func, None, True)

def _findMaximizeDirection2D(x, function, currentLoss, dx):
	xChanges = [x + dx, x - dx]
	lossChanges = [function(i) for i in xChanges]
	directionToGo = xChanges[lossChanges.index(max(lossChanges))]
	return directionToGo

def maximize2D(func, goal, startx=np.random.rand(1)[0], dx=0.01, maxIter=3000, calculateMovement=True, verbose=False):
	iteration = 0
	currentX = startx
	currentLoss = func(currentX)
	historyX = []
	historyY = []

	while iteration < maxIter:
		if verbose:
			print('Iteration:', iteration, 'X:', currentX, 'Loss:', currentLoss)
		if iteration % 20 == 0:
			historyX.append(currentX)
			historyY.append(currentLoss)
		# Take step in x direction, and calculate loss increase/decrease (not slope because we don't care about magnitude), then find direction to move in
		if calculateMovement:
			movementAmountX = np.sqrt(abs(currentLoss)) * dx / 2
			moveDir = _findMaximizeDirection2D(currentX, func, currentLoss, movementAmountX)
		else:
			moveDir = _findMaximizeDirection2D(currentX, func, currentLoss, dx)
		# Move in that direction
		newX = moveDir
		currentX = newX
		# Recalculate loss
		newLoss = func(currentX)
		currentLoss = newLoss
		if currentLoss >= goal:
			break
		iteration += 1

	return MinimizerOutput((currentX), currentLoss, ((historyX, historyY), iteration), func, None, True)

# Testing the system

if __name__ == "__main__":
	pass

	# # Example 1 - Linear functions for finding the correlation of an mx + b function.
	# X = np.array([1, 2])
	# y = np.array([4, 7])
	# 
	# # visualizeLoss(X, y, linear)
	# output = fit(loss, X, y, startx=np.random.rand(1)[0], starty=np.random.rand(1)[0], dx=0.1, dy=0.1, maxIter=3000, lossThresh=0.000001, verbose=False)
	# print(output.answers, output.loss, output.iterations)
	# output.plotLossChart()

	# # Example 2 - Quadratic functions for finding the correlation of an mx^2 + bx function.
	# X = np.array(frange(-5, 5, 0.1))
	# y = np.array(quadratic(3, 2, X))
	# 
	# # visualizeLoss(X, y, quadratic)
	# output = fit(loss, X, y, func=quadratic, startx=np.random.rand(1)[0], starty=np.random.rand(1)[0], dx=0.01, dy=0.01, maxIter=3000, lossThresh=0.005, verbose=False)
	# print(output.answers, output.loss, output.iterations)
	# output.plotFitting()

	# # Example 3 - Custom Optimizations using a^2 + b^2 = 25 and a + b = 7
	# def customLoss(xval, yval, X=None, y=None, func=None):
	# 	lossA = (xval ** 2 + yval ** 2) - 25
	# 	lossB = (xval + yval) - 7
	# 	return np.sum(np.square(np.array([lossA, lossB]))) / 2 
	# 
	# visualizeLoss(None, None, lossFunc=customLoss)
	# output = fit(None, None, None, maxIter=4000, lossThresh=0.0001)
	# print(output.answers, output.loss, output.iterations)
	# output.plotLossChart()

	# # Example 4 - Function minimization using Himmelblau's Function (https://en.wikipedia.org/wiki/Himmelblau%27s_function)
	# def himmelblau(x, y):
	# 	return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
	# 
	# visualizeFunction(himmelblau, -5, -5, 5, 5, 0.5, 0.5)
	# output = minimize(himmelblau, 0.01)
	# print(output.answers, output.loss, output.iterations, output.finished)
	# output.plotLossChart()

	# # Example 5 - Function maximization using -x^2 - y^2 + 100
	# def maxFunc(x, y):
	# 	return -(x-1)**2 - y**2 + 100
	# 
	# visualizeFunction(maxFunc, -5, -5, 5, 5, 0.5, 0.5)
	# output = maximize(maxFunc, 99.9999999)
	# print(output.answers, output.loss, output.iterations, output.finished)
	# output.plotLossChart()

	# # Example 6 - Function minimization with non-differentiable equations
	# def minFunc(x, y):
	# 	return abs(x - 1) + abs(y + 1)
	# 
	# visualizeFunction(minFunc)
	# output = minimize(minFunc, 0.01)
	# print(output.answers, output.loss, output.iterations, output.finished)
	# output.plotLossChart()

	# # Example 7 - Function minimization using the McCormick function (https://www.sfu.ca/~ssurjano/mccorm.html)
	# def mccormick(x, y):
	# 	return math.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1
	# 
	# visualizeFunction(mccormick, -5, -5, 5, 5, 0.5, 0.5)
	# output = minimize(mccormick, -1.9133)
	# print(output.answers, output.loss, output.iterations, output.finished)
	# output.plotLossChart()

	# # Example 8 - Single variable optimization
	# # Question: Given a rectangle of area 25, minimize the lengths of sides to get the least perimeter.
	# def rectLoss(x):
	# 	if x == 0:
	# 		return 0
	# 	x = abs(x)
	# 	side1 = x
	# 	side2 = 25/x
	# 	return 2 * (side1 + side2)
	# 
	# output = minimize2D(rectLoss, 20, maxIter=5000)
	# print(output.answers, output.loss, output.iterations, output.finished)

	# # Example 9 - Example with word problem.
	# # Question: Two people have ages with a sum of 66. One person's age is the other person's age backwards.
	# def ageLoss(x, y):
	# 	sumOfAges = 11 * x + 11 * y
	# 	return ((sumOfAges - 66) ** 2) / 2
	# 
	# output = minimize(ageLoss, 0, startx=random.randint(5, 9), starty=random.randint(6, 9), dx=1, dy=1, maxIter=5000, calculateMovement=False)
	# print(output.answers, output.loss, output.iterations)
	# output.plotLossChart()
