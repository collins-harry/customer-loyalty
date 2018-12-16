import numpy as np

# initial probability of coin 0 or 1
#pi = (1/2, 1/2)
pi = (.4, .6)
#transition probabilities P(state0->state1) = A[0][1]
#A = np.array([[.4, .6], [.6, .4]])
A = np.array([[.4, .6], [.7, .3]])
#emission probabilities (coin x prob(H and T)
#B = np.array([[.3, .7], [.8, .2]])
B = np.array([[.3, .7], [.6, .4]])

sequence = [0, 1, 0, 1]

#######################Initial \alpha#####################
alpha = []
#alpha is observation x latent variable
alpha.append(np.array([[pi[i]*B[i][j] for i in range(2)] for j in range(2)]))
seq = 0
print(f'alpha_0\n', alpha[0], '\n')

######forward pass#######


for index, seq in enumerate(sequence):

    alpha[index][seq] = [0, 0]
    #print(f'since sequence is T\n', alpha[index], '\n')

    alpha[index] = alpha[index].sum(axis=0)
    #print(f'sum over obs in alpha_{index} -> prob of coin1 and coin2 in t\n', alpha[index], '\n')

    temp1 = np.matmul(alpha[index], A)
    #print(f'alpha_{index} dot A -> prob of coin1 and coin2 in t+1\n', temp1, '\n')


    alpha.append(np.array([[temp1[i]*B[i][j] for i in range(2)] for j in range(2)]))
    #print(f'alpha_{index+1}, multiplying by B)\n', alpha[index+1], '\n')

for i in range(4):
    print(alpha[i])

print('=========backward pass========='

beta = []
beta.append(1)

temp1 = A.dot(B)
print('\n A dot B\n',
