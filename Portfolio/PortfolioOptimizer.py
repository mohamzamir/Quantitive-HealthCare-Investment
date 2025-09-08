import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

col_names = ['Name', 'Risk', 'Return']

np.random.seed(42)

# gendata = True
# if gendata == True:
#     #cell to generate random data
#     rand_data = []
#     num_pts = 10

#     mean = 10 # average log normal return
#     returns_noise = 0.5 # log normal returns

#     for i in range(num_pts):
#         risk = 1 - np.max(1 / (mean + np.random.normal()), 0)
#         rand_data.append(["Disease " + str(i), risk, np.exp(np.log(mean) + returns_noise * np.random.normal())])
        
#     rand_data_df = pd.DataFrame(rand_data, columns = col_names)
#     rand_data_df.to_csv("./riskreturn.csv", index=False)


input = "./predicted.csv"
data = pd.read_csv(input, nrows=100)
n = len(data)
# CovarRisk = np.diag(data['Risk'] ** 2)

# MeanReturns = data['Return']
# AEq = np.ones(MeanReturns.T.shape)

# def MaximizeReturns(data, PortfolioSize): #optimize with no risk
#     c = np.multiply(-1, data['Return'].tolist())
#     A = np.ones([PortfolioSize, 1]).T
#     b = [1]
#     weights = opt.linprog(c, A_ub = A, b_ub = b, bounds = (0,1))
#     return weights.x

# def MinimizeReturns(data, size):
#     def f(x):
#         return np.matmul(np.matmul(x, CovarRisk), x.T)

#     def constraintEq(x):
#         return np.matmul(AEq,x.T)-1 
    
#     xinit=np.repeat(0.1, size)
#     cons = ({'type': 'eq', 'fun':constraintEq})
#     lb = 0
#     ub = 1
#     bnds = tuple([(lb,ub) for x in xinit])

#     weights = opt.minimize (f, x0 = xinit,  bounds = bnds, \
#                              constraints = cons, tol = 0.001)
    
#     return weights.x

# expected_mu = data['Return'] * (1-data['Risk'])

# def MeanVar(data, size, R):    
#     def f(x):
#         return -np.matmul(expected_mu, x.T) 

#     def constraintEq(x):
#         return np.matmul(AEq,x.T)-1 
    
#     def constraintIneq(x, R):
#         return np.matmul(data['Risk'], x.T) - R

#     xinit=np.repeat(1/size, size)
#     cons = ({'type': 'eq', 'fun':constraintEq},
#             {'type':'ineq', 'fun':constraintIneq, 'args': [R] })
#     lb = 0
#     ub = 1
#     bnds = tuple([(lb,ub) for x in xinit])

#     res = opt.minimize (f, x0 = xinit, bounds = bnds, constraints = cons, tol = 0.001)
#     return res.x

def ComputeReturns(data, weights):
    return np.matmul(np.array(data['Return'] * (1 - data['Risk']) - data['Risk']), weights.T)

# # def ComputeRisk(data, weights):
# #     return np.sqrt(np.matmul(np.matmul(weights, CovarRisk), weights.T))

# # n = len(data)
# incr = 0.01

# # min = ComputeReturns(data, MinimizeReturns(data, n))
# # max = ComputeReturns(data, MaximizeReturns(data, n))

# # print(min)
# # print(max)

# frontier_risk = []
# frontier_return = []
# frontier_weights = []

# R = 0 # target return
# while R < 1:
#     res = MeanVar(data, n, R)
#     frontier_risk.append(R)
#     frontier_return.append(ComputeReturns(data, res))
#     frontier_weights.append(res)
#     R += incr
    
    # print(R)\
        
from pypfopt.efficient_frontier import EfficientFrontier
import pypfopt.objective_functions as objective_functions

print("pypfopt")
expected_returns = data['Return'] * (1 - data['Risk'])
# cov_matrix = np.array([[(data['Risk'][i]**2 if i == j else 0) for i in range(n)] for j in range(n)])
risk = [(data['Return'][i]**2 * (1-data['Risk'][i]) + data['Risk'][i]) - (data['Return'][i] * (1-data['Risk'][i]) - data['Risk'][i])**2 for i in range(n)]
cov_matrix = np.diag(risk)
print(cov_matrix)
# print(cov_matrix)
print("t1")

ef = EfficientFrontier(expected_returns, cov_matrix)
ef.add_objective(objective_functions.L2_reg, gamma = 0.1)

print("t2")
n = len(data)
incr = 0.01
min = 0.0
max = 5

print(min)
print(max)

frontier_risk = []
frontier_return = []
frontier_weights = []
R = min # risk tolerance
tol = 0.000001
prev = 0.0
print("t3")
itr = 0
while R < max:
    try:
        res = ef.efficient_risk(R*R)
        print("t4")
        res = np.array([i[1] for i in res.items()])
        
        print("t5")
        frontier_risk.append(round(R, 2))
        frontier_return.append(ComputeReturns(data, res))
        frontier_weights.append(res)
        R += incr
        # if frontier_return[-1] - prev < tol:
        #     break
        prev = frontier_return[-1]
        itr += 1
        if itr % 10 == 0:
            pd.DataFrame(frontier_weights, columns = [str(i) for i in range(n)]).to_csv("./weights.csv", index = False)
            pd.DataFrame(zip(frontier_risk,frontier_return), columns = ['Risk', 'Return']).to_csv("./efficient_frontier.csv", index = False)
    except ValueError as e:
        print("except")
        R += incr
    
plt.figure()
plt.scatter(frontier_risk, frontier_return)
plt.xlabel("Risk (%)")
plt.ylabel("Expected Return (%)")
plt.show()

pd.DataFrame(frontier_weights, columns = [str(i) for i in range(n)]).to_csv("./weights.csv", index = False)
pd.DataFrame(zip(frontier_risk,frontier_return), columns = ['Risk', 'Return']).to_csv("./efficient_frontier.csv", index = False)