import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom

mu = (1/3,1/2)
pi = (1/2,1/2)

x = [(1,4),(3,2),(4,1),(2,3)]
x_test = [(x, 5-x) for x in range(6)]
#probs_x_given_mu = dict()
#probs_x = dict()

def calc_gammas(pi, mu, x):
    gammas = dict()
    for h,t in x: 
        prob_x_given_mu = tuple(ncr(h+t, h) * mu[k]**h * (1-mu[k])**t for k in range(2))
        #probs_x_given_mu[(h,t)] = prob_x_given_mu

        prob_x = sum([pi[k]*prob_x_given_mu[k] for k in range(2)])
        #probs_x[(h,t)] = prob_x

        gammas[(h,t)] = tuple(pi[k] * prob_x_given_mu[k] / prob_x for k in range(2))
    return gammas

gammas = calc_gammas(pi, mu, x)

print("Exercise 1 part a") 
print("---------------------------")
print('        gamma(Zn1)           gamma(Zn2)')
for result , gamma in gammas.items():
    print(result, gamma)
print('')

print("Exercise 1 part b")
print("---------------------------")
Workings = '''
In order to get the new optimal mu we need to set the derivative of the expected value of log likelihood function with respect found latent variable distributions (the coin responsibilities)

This results in mu = 1/sum(responsibilities_k) * sum(responsibilities_k)*x     where the sums sum over each data point.
'''
print(Workings)
print('')
print('calculating mu where we use the probabilities of the responsibilities instead of the binary prediction')
N = dict()
N[0] = sum([resp_1 for resp_1, resp_2 in gammas.values()])
N[1] = sum([resp_2 for resp_1, resp_2 in gammas.values()])

#temp1, temp2 = gammas[(2, 3)]
#N[0] += temp1
#N[1] += temp2
print('')
print('estimated number of times coin 1 and 2 were used:')
print('')
print('     N_0                      N_1')
print('---------------------------------------------')
print(N[0],'      ',  N[1])
print('')

X = dict()
mean_sum = [0,0] 
for h, t in x:
    resp_coin_1, resp_coin_2 = gammas[(h, t)]
    mean_sum[0] += resp_coin_1 * h/(h+t)
    mean_sum[1] += resp_coin_2 * h/(h+t)

new_mu = [mean_sum[i] / N[i] for i in range(2)]    

print('           new mu values')
print('         -----------------')
print('coin_1 prob_heads    coin_2 prob_head')
print('------------------------------------')
print(new_mu)
print('')


new_pi = [N[i]/sum(N.values()) for i in range(2)]
print('         new pi values')
print('       -----------------')
print('coin_1 prob_use      coin_2 prob_use')
print('------------------------------------')
print(new_pi)
print('')








