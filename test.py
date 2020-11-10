from main import *

# X = np.array([1, 2, 3, 4, 5])
# y = np.array([3, 5, 7, 9, 11])
# 
# # X = 2y + 1
# 
# # visualizeLoss(X, y, linear)
# output = fit(loss, X, y, func=linear, lossThresh=0.0001)
# print(output.answers, output.loss, output.iterations)
# print(findAnswer(output, 7))
# 
# output.plotFitting()

# X = np.array([i for i in range(-5, 5)])
# y = np.array([3 * i ** 2 + 2 * i for i in X])
# 
# # 3x^2 + 2x
# 
# # visualizeLoss(X, y, quadratic)
# output = fit(loss, X, y, func=quadratic, lossThresh=0.0001)
# print(output.answers, output.loss, output.iterations)
# output.plotLossMovement()

# # x^2 + y^2 = 25
# # x + y = 7
# 
# def customLoss(xval, yval, X=None, y=None, func=None):
# 	return ((((xval ** 2 + yval ** 2) - 25) ** 2 + ((xval + yval) - 7) ** 2)) / 2
# 
# output = fit(customLoss, None, None, func=None, lossThresh=0.01)
# print(output.answers, output.loss, output.iterations)
# print(output.answers[0] + output.answers[1])

# # A window is being built and the bottom is a rectangle and the top is a semicircle. If there is 12 meters of framing materials what must the dimensions of the window be to let in the most light?
# 
# def toMaximize(r):
# 	return 12 * r - 2 * r ** 2 - (1/2) * math.pi * r ** 2
# 
# visualize2D(toMaximize)
# 
# output = maximize2D(toMaximize, 20)
# print(output.answers, output.loss, output.iterations)
