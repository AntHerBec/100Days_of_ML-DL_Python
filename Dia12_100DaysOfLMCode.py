
### Naive Bayes Classifiers ###

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y = True) # Directly return X and y from the iris dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3,
                                                   shuffle = True, stratify = y)

## Since there isn't an easy multivariate gaussian goodness of fit test, we'll use kolmogorov-smirnov
## with a chi^2 distribution that comes from using mahalanobis distance (basically the formula for
## standardising a normal distribution, substract the mean and multiply by the variance inverse matrix)

nsamples, nfeatures = X.shape
mean = np.mean(X, axis = 0)
var = np.cov(X, rowvar = False)
inv_var = np.linalg.inv(var)

## we sum the distances
diff = X - mean
print(mean)
print(diff)
print(diff @ inv_var @ diff.T)
md_sq = np.sum(diff @ inv_var * diff, axis = 1)

result = sts.kstest(md_sq, "chi2", args = (nfeatures,))

print(f"KS Statistic: {result.statistic:.4f}")
print(f"P-value:      {result.pvalue:.4f}")
print(f"Normal?       {'Yes' if result.pvalue > 0.05 else 'No'}")

# we don't have enough evidence to deny the data behaving like a multi-gaussian, so we can now
# apply the gaussian naive bayes classifier and see what results we obtain

gnb = GaussianNB()
gnb.fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(
    gnb,
    X_test,
    y_test,
    display_labels = ["Setosa", "Versicolour", "Virginica"]
)
plt.show()

# The results are pretty good, with some problems with Versicolour and Virginica because both samples are
# overlapping, causing the algorithm to misclassify some of them. The same could be applied to other
# distributions, such as the multinomial. It must be kept in mind that this method goes under the assumption
# that the different features are independent, so under considerable correlation we shouldn't proceed this way














