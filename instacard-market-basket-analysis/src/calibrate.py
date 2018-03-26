import sys
import json
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.isotonic
from random import shuffle

data = []
for line in sys.stdin:
    rec = json.loads(line)
    for item in rec:
        data.append((int(float(item[0])), float(item[1])))
data.sort(key=lambda x: x[1])

bucket_num = int(sys.argv[1])
labels = []
scores = []
current_score = '-1.0'
label_in_same_score = []
for (label, score) in data:
    if score != current_score and current_score != '-1.0':
        shuffle(label_in_same_score)
        for l in label_in_same_score:
            labels.append(int(float(l)))
            scores.append(float(current_score))
        label_in_same_score = []
    label_in_same_score.append(label)
    current_score = score
for l in label_in_same_score:
    labels.append(int(float(l)))
    scores.append(float(current_score))
step_size = len(labels) / bucket_num

totals = []
positives = []
ectrs = []
boundaries = [0.0]
total_in_bucket = 0
positive_in_bucket = 0
for i in range(0, len(labels)):
    if i % step_size == 0 and i != 0:
        ectr = float(positive_in_bucket) / total_in_bucket
        totals.append(total_in_bucket)
        positives.append(positive_in_bucket)
        ectrs.append(ectr)
        boundaries.append(scores[i])
        total_in_bucket = 0
        positive_in_bucket = 0
    total_in_bucket += 1
    positive_in_bucket += labels[i]
if total_in_bucket < step_size / 4:
    total_in_bucket += totals.pop()
    positive_in_bucket += positives.pop()
    ectrs[-1] = float(positive_in_bucket) / total_in_bucket
    boundaries.pop()
else:
    ectr = float(positive_in_bucket) / total_in_bucket
    totals.append(total_in_bucket)
    positives.append(positive_in_bucket)
    ectrs.append(ectr)

ir = sklearn.isotonic.IsotonicRegression(increasing='auto')
ir.fit(boundaries, ectrs)
fitted_ctrs = ir.predict(boundaries)

#print ','.join(map(str,positives))
#print ','.join(map(str,totals))

print ','.join(map(str,boundaries))
print ','.join(map(str,fitted_ctrs))
