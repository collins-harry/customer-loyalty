import numpy as np

# initial probability of coin 0 or 1
pi = (1/2, 1/2)
#transition probabilities P(state0->state1) = A[0][1]
A = np.array([[.4, .6], [.6, .4]])
#emission probabilities (coin x prob(H and T)
B = np.array([[.3, .7], [.8, .2]])

sequence1 = [0, 0, 1, 1]
print('===========exercise 2.1==========')
#######################Initial \alpha#####################
def alpha_beta_calc(sequence, A, B):
    alpha = []
    #alpha is observation x latent variable
    alpha.append(np.array([[pi[i]*B[i][j] for i in range(2)] for j in range(2)]))
    seq = 0
    print(f'alpha_0\n', alpha[0], '\n')

    ######forward pass#######

    print('---------forward pass-----------')

    for index, seq in enumerate(sequence):

        alpha[index][seq] = [0, 0]
        #print(f'since sequence is T\n', alpha[index], '\n')

        alpha[index] = alpha[index].sum(axis=0)
        #print(f'sum over obs in alpha_{index} -> prob of coin1 and coin2 in t\n', alpha[index], '\n')
        #print(alpha[index])

        temp1 = np.matmul(alpha[index], A)
        #print(f'alpha_{index} dot A -> prob of coin1 and coin2 in t+1\n', temp1, '\n')


        alpha.append(np.array([[temp1[i]*B[i][j] for i in range(2)] for j in range(2)]))
        #print(f'alpha_{index+1}, multiplying by B)\n', alpha[index+1], '\n')

    for i in range(4):
        print(alpha[i])

    print('\n----------backward pass----------')

    print('\nA\n',A)
    print('\nB\n',B)
    beta = []
    beta.append(np.array([1,1]))
    for index, seq in enumerate(reversed(sequence)):
        seq = seq - 1
        tempB = np.array([B[:, seq ],B[:, seq ]])
        #print('\nB\n',B)
        #print('\ntempB\n',tempB)
        temp1 = A * tempB
        #print('\n A dot tempB\n',temp1)
        #print(f'\n beta{index}\n',beta[index])
        #beta[index][:,seq] = [0, 0]
        tempBeta = np.array([beta[index], beta[index]])
        beta.append((temp1 * tempBeta).sum(axis=1))
        #print(f'\n beta{index+1}\n',beta[index+1])
        #beta[index][:, seq] = [0,0]
        #print(f'\n bet {index}\n',beta[index])

    print('idx  beta')
    for index, bet in enumerate(reversed(beta[:4])):
        print(4-index,': ', bet)

    return alpha, beta

alpha, beta = alpha_beta_calc(sequence1, A, B)
print('========exercise 2.2============')

print('sum over all alpha_4 (note it is actually a square matrix)')
print('alpha_4,1   alpha_4,2 ')
print(alpha[3])
answer = alpha[3].sum() 
print(f'P(O|M) = {answer}')

print('\n========exercise 2.3============\n')
observ1 = answer

print('--------gamma1---------')
beta = list(reversed(beta[:4]))

def calc_gamma(alpha, beta, observ):
    gamma = []
    for i in range(4):
        temp = [alpha[i][j]*beta[i][j]/observ for j in range(2)]
        gamma.append(temp)
    return gamma

gamma1 = calc_gamma(alpha, beta, observ1)

for index, gamma in enumerate(gamma1):
    print(f'gamma{index+1}: ', gamma)

print('\n-------Eta1-------')

def calc_eta(A, B, alpha, beta, sequence, observ):
    eta = []
    for idx, seq in enumerate(sequence[1:]):
        seq = 1 - seq
        temp = B[:,seq] * beta[idx+1]  
        temp2 = np.array([temp, temp])
        temp3 = temp2 * A
        tempAlpha = np.array([alpha[idx], alpha[idx]]).T
        eta.append(tempAlpha * temp3 / observ)
    return eta
        
eta1 = calc_eta(A, B, alpha, beta, sequence1, observ1)
for idx, eta in enumerate(eta1):
    print(f'eta{idx+1}: \n{eta}')


print('--------gamma2---------')

#gamma2 = calc_gamma(alpha, beta, observ)

print('--------eta2---------')








