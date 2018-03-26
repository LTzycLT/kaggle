import sys
sys.path.append(".")
import json
import numpy as np
from utils import calibration

def get_best_products(r):
    r.sort(key=lambda o: o[1], reverse=True)
    labels = np.array([rr[0] for rr in r])
    probs =  np.array([calibration(rr[1]) for rr in r])
#    probs =  np.array([rr[1] for rr in r])
    product_ids =  np.array([rr[2] for rr in r])
#    k, predNone, predicted_f1 = maximize_expectation(probs)
    k, predNone, predicted_f1 = choose(probs)
    best_products = np.append(product_ids[:k], ["None"]) if predNone else product_ids[:k]
    true_f1 = f1_score(labels, k, predNone)
    return best_products, true_f1

def f1_score(labels, k, predNone=False):
    if sum(labels) > 0 and k > 0:
        p = sum(labels[:k])/(k+1 if predNone else k)
        r = sum(labels[:k])/sum(labels)
        if p+r > 0: return 2*p*r/(p+r)
    if sum(labels) == 0 and predNone:
        p = 1/(k+1)
        r = 1
        return 2*p*r/(p+r)
    return 0


# The naive algo.
def choose(probs, pNone=None):
    # Kept this sketch to adopt f1_predict(), though it is not necessary for f1_predict2(). For f1_predict2(), whenever k > 0, predNone is defined to be False.
    max_score, j, predNone = -1, -1, False
    for i in range(len(probs) + 1):
#        score = f1_predict(probs, i, predNone=False)
        score = f1_predict2(probs, i, predNone=False)
        if score > max_score:
            max_score = score
            j = i
            predNone = False
    for i in range(len(probs) + 1):
#        score = f1_predict(probs, i, predNone=True)
        score = f1_predict2(probs, i, predNone=True)
        if score > max_score:
            max_score = score
            j = i
            predNone = True
    return j, predNone, max_score

def f1_predict(probs, k, predNone=False):
    # This algo is literally incorrect. Because "None" is treated as an ordinary product but the existence of "None" is not independent of the existense of products.
    # It is left here just for further investigations.
    pNone = (1-probs).prod()
    if k == 0 and predNone:
        return pNone
    tp = probs[:k].sum() + pNone if predNone else probs[:k].sum()
    p = tp/(k+1 if predNone else k)
    r = tp/(probs.sum() + pNone if predNone else probs.sum())
    return 2*p*r /(p+r) if p+r > 0 else 0

def f1_predict2(probs, k, predNone=False):
    pNone = (1-probs).prod()
    if k == 0: return pNone if predNone else 0
    tp = probs[:k].sum()
    p = tp/k
    r = tp/probs.sum()
    return 2*p*r /(p+r) if p+r > 0 else 0


# The O(n^2) algo. 
def maximize_expectation(P, pNone=None):
    expectations = get_expectations(P, pNone)
    ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
    max_f1 = expectations[ix_max]
    predNone = True if ix_max[0] == 0 else False
    best_k = ix_max[1]
    return best_k, predNone, max_f1

def get_expectations(P, pNone=None):
    expectations = []
    P = np.sort(P)[::-1]

    n = np.array(P).shape[0]
    DP_C = np.zeros((n + 2, n + 1))
    if pNone is None:
        pNone = (1.0 - P).prod()

    DP_C[0][0] = 1.0
    for j in range(1, n):
        DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

    for i in range(1, n + 1):
        DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
        for j in range(i + 1, n + 1):
            DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

    DP_S = np.zeros((2 * n + 1,))
    DP_SNone = np.zeros((2 * n + 1,))
    for i in range(1, 2 * n + 1):
        DP_S[i] = 1. / (1. * i)
        DP_SNone[i] = 1. / (1. * i + 1)
    for k in range(n + 1)[::-1]:
        f1 = 0
        f1None = 0
        for k1 in range(n + 1):
            f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
            f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
        for i in range(1, 2 * k - 1):
            DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
            DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
        expectations.append([f1None + 2 * pNone / (2 + k), f1])

    return np.array(expectations[::-1]).T




if __name__ == "__main__":
    for line in sys.stdin:
        rec = json.loads(line)
        (prods, score) = get_best_products(rec)
        print score
