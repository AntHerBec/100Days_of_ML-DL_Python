
import numpy as np
import random as r
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import ConfusionMatrixDisplay


## Let's consider a dataset where we have 3 tricked dices, each with different probabilities assigned to each
## number. We can generate it using the "random" module, considering 5 different variables (the result of
## 5 throws) so that the dices 1,2 and 3 are our classes

t0 = time.time()

p = [[0.1, 0.2, 0.15, 0.05, 0.3, 0.2], [0.14, 0.17, 0.17, 0.16, 0.14, 0.22], [0.18, 0.15, 0.18, 0.17, 0.16, 0.16]]
X, y = [], []

for i in range(500):
    X1 = np.array(r.choices(population = np.linspace(1,6,6), weights = p[0], k = 300))
    X2 = np.array(r.choices(population = np.linspace(1,6,6), weights = p[1], k = 300))
    X3 = np.array(r.choices(population = np.linspace(1,6,6), weights = p[2], k = 300))
    Xlist = [X1, X2, X3]
    i = 0
    for k in Xlist:
        counts = []
        for j in range(1,7):
            counts.append(len(k[np.where(k == j)]))
        y.append(i)
        X.append(counts)
        i += 1

X = np.array(X)
y = np.array(y)

data = np.concatenate((X,y.reshape(-1,1)), axis = 1)
np.random.shuffle(data)

## Since it comes from a designed dataset simulating dices, we know for a fact that it behaves like a
## multinomial distribution. Therefore, we can apply the MultinomialNB and check the results

X, y = data[:,:-1], data[:,-1]
mnb = MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3,
                                                   shuffle = True, stratify = y)
mnb.fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(
    mnb,
    X_test,
    y_test,
    display_labels = ["Dice 1", "Dice 2", "Dice 3"]
)
plt.show()

# We can determine that the data classification is quite accurate, with (75-80)% of the data being correctly
# labeled. This result can be easily improved by increasing the sample number from each class (although we
# will always have a considerable amount of missclasification due to only throwing dices 2 and 3 100 times,
# since their probabilities don't differ much) or more effectively increasing the number of dice throws (this
# makes much clearer the differences and the analysis from NB works perfectly, almost getting completely right
# results in the test data)
#
# For the purpose of representing this same study in a visual way, we will now emulate the throw of 3 coins, 2
# tricked and 1 non-tricked coin, and represent them. we will apply the np.random.multinomial function for both
# variety and because I feel a little dumb for overcomplicating everything >:(

p = [[0.5, 0.5], [0.4, 0.6], [0.65, 0.35]]
nrolls, nsamples = [100, 100]

X1 = np.random.multinomial(nrolls, p[0], size = nsamples)
print(X1)
X2 = np.random.multinomial(nrolls, p[1], size = nsamples)
X3 = np.random.multinomial(nrolls, p[2], size = nsamples)

X = np.vstack([X1,X2,X3])
y = np.concatenate([np.zeros(nsamples), np.ones(nsamples), np.full(nsamples, 2)])

mnb = MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3,
                                                   shuffle = True, stratify = y)
mnb.fit(X_train, y_train)
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16,8))
DecisionBoundaryDisplay.from_estimator(
    mnb,
    X,                          # we want to check how the totality of the data fits the created boundaries
    response_method="predict",
    plot_method="pcolormesh",
    shading = "auto",
    alpha = 0.5,
    ax = axes[0]
)
axes[0].scatter(X[y == 0, 0], X[y == 0, 1], edgecolors = "k")
axes[0].scatter(X[y == 1, 0], X[y == 1, 1], edgecolors = "k")
axes[0].scatter(X[y == 2., 0], X[y == 2., 1], edgecolors = "k")
ConfusionMatrixDisplay.from_estimator(
    mnb,
    X_test,
    y_test,
    display_labels = ["Coin 1", "Coin 2", "Coin 3"],
    ax = axes[1]
)
plt.tight_layout()
plt.show()

