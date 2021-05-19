
# roc curve and auc
from matplotlib import pyplot
from sklearn import metrics
import numpy as np


x1, y1 = [0, 1], [0, 1]
pyplot.plot(x1, y1,linestyle='--')


#y = np.array([0, 1, 1, 0])
y = []
y.append(0)
y.append(0)
y.append(0)
y.append(1)
y = np.array(y)
print(y)
scores = []
scores.append(0.9)
scores.append(0.95)
scores.append(0.98)
scores.append(0.9)
scores = np.array(scores)
#scores = np.array([0.2, 0.4, 0.6, 0.8])
print(scores)


fpr, tpr, thresholds = metrics.roc_curve(y, scores)
auc = metrics.roc_auc_score(y, scores)
print(thresholds)

pyplot.plot(fpr, tpr, marker='.', label='ResNet AUC = '+str(auc))
pyplot.xlabel('1 - especificidade')
pyplot.ylabel('Sensibilidade')
pyplot.legend(loc=4)
pyplot.show()