import pickle
import sys

with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

def np2csv(a,c):
    f = open(c, "w")
    if a.ndim == 2:
        for r in a:
            f.write(','.join([str(x) for x in r.tolist()]) + '\n')
    elif a.ndim == 1:
        f.write(','.join([str(x) for x in a.tolist()]) + '\n')
    else:
        print("data error")
        sys.exit(1)
    f.close()

np2csv(network['W1'], "sample_weight_W1.csv")
np2csv(network['W2'], "sample_weight_W2.csv")
np2csv(network['W3'], "sample_weight_W3.csv")

np2csv(network['b1'], "sample_weight_b1.csv")
#np2csv(network['b2'], "sample_weight_b2.csv")
#np2csv(network['b3'], "sample_weight_b3.csv")

