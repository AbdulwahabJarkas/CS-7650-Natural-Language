from doctest import OutputChecker
import numpy as np

attention = np.array([1,2,4]).reshape(1, 3)
_is = np.array([-1,0,2]).reshape(1, 3)
all = np.array([3,1,3]).reshape(1, 3)
you = np.array([5,0,0]).reshape(1, 3)
need = np.array([2,-2,-1]).reshape(1, 3)

X = np.vstack((attention, _is, all, you , need))

WQ = np.array([[1,1],[-3,1],[-2,3]])
WK = np.array([[-1,3],[-2,-5],[-1,-2]])
WV = np.array([[3,0],[2,-4],[4,0]])


# q1, q2, q3, q4, q5 = attention @ WQ, _is @ WQ, all @ WQ, you @ WQ, need @ WQ
# k1, k2, k3, k4, k5 = attention @ WK, _is @ WK, all @ WK, you @ WK, need @ WK
# v1, v2, v3, v4, v5 = attention @ WV, _is @ WV, all @ WV, you @ WV, need @ WV

# Part 1
Q, K, V = np.dot(X, WQ), np.dot(X, WK), np.dot(X, WV)

normalized_scores = np.dot(Q, K.T)/48
#print(normalized_scores)

# Part 2
def softmax(array):
    return np.exp(array)/np.sum(np.exp(array), axis=1, keepdims=True)

attention_values = softmax(normalized_scores)
#print(attention_values)

# Part 3
outputs1 = np.dot(attention_values, V)
#print(outputs1)

# # Part 4
outputs2 = np.array([[1.5, -2.5], [-4.78, 0.15], [1.75, -1.97], [-3.96, -2.9], [-0.53, 4.61]])
outputs3 = np.array([[1.51, 0.07], [-4.95, -3.47], [2.33, -4.81], [0.05, 0.68], [2.85, -1.91]])
outputs4 = np.array([[-3.59, -3.18], [3.38, -1.85], [3.77, 4.21], [-0.15, 1.46], [-1.65, 1.51]])

outputs = np.hstack((outputs1, outputs2, outputs3, outputs4))

WO = np.array([[-5, 4, -5], [2, -1, 2], [1, -4, 4], [0, 0, 0], [3, 3, -1], [-4, -3, -1], [3, 1, 5], [1, 3, 3]])

self_attention_output = np.dot(outputs, WO)
print(self_attention_output)