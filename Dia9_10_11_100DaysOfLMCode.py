
'''
Today's practice started by understanding the way in which kd-trees actually work. After learning of the
algorithm visually (props to wikipedia's video) I still had some doubts about how can the algorithm find
the second, third and so on nearest neighbors, since the pages I had looked at all mentioned that only
1 application of the algorithm was enough. In the end, the idea was very simple: getting a list with the
 first k neighbours available, and just compare the new ones to the worst of them all. If it is closer, it
goes in while the furthest one goes out.
'''

### SVM start ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale # standardize a dataset in any axis
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_gaussian_quantiles
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV # does cross validation
from sklearn.svm import SVC # Classifier made from support vector machine
from sklearn.decomposition import PCA # To reduce the dimensionality of data for plotting
from sklearn.metrics import ConfusionMatrixDisplay # Shows visually the confusion matrix
import seaborn as sns
from Repertorio_Funciones import *


df = pd.read_excel("default of credit card clients.xls")
print(df.head(5).to_markdown())

sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = "viridis")
plt.show()

# There aren't any null values. However, there are unique values in some columns that ar not registered in the
# information below, so if the don't represent a considerable amount of the data we could delete it

for i in range(len(df.iloc[0,:].values)):
    print("unique values of {}: {}".format(df.iloc[0,i], np.sort(df.iloc[1:,i].unique())))

# Let's get rid of the first row, and use as column names the second one
df = df.rename(columns = df.iloc[0,:]).iloc[1:,:]
print(df.head(5).to_markdown())

'''
Now we explain the values of each variable:

LIMIT_BAL: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit

SEX: 1 = male 2 = female EDUCATION: 1 = graduate school 2 = university 3 = high school 4 = others MARRIAGE: 1 = married 2 = single 3 = others AGE: Age (year)

PAY_: -1 = pay duly 1 = payment delay for one month 2 = payment delay for two months . . . 8 = payment delay for eight months 9 = payment delay for nine months and above

BILL_AMT: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .;
X17 = amount of bill statement in April, 2005

PAY_AMT: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005

DEFAULT: Defaulted = 1 Did not default = 0
'''

# It can be assumed that 5 and 6 in "EDUCATION" might be extra categories that weren't explained. Also, in the last one
# -2 is not known either but it might refer to payment beforehand. However, the meaning of 0 in "MARRIAGE" "EDUCATION"
# and "PAY_" are more difficult to think of, and it can lead to missing_values interpretation from the algorithm. For
# that, we must get rid of them (if the number of zeros in a column is much lower than the total length)

zero_columns = ["EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
for i in zero_columns:
    print("Number of zeros in {}: ".format(i), len(df.loc[(df[i].isin([0]))]))

# Since "PAY_" zeros are very frecuent, we will only delete the rows with zeros on "MARRIAGE" or "EDUCATION"
zeroindex = df[(df["MARRIAGE"] == 0) | (df["EDUCATION"] == 0)].index
df = df.drop(zeroindex)

# Are the values of y (0 and 1) similarly frecuent? How big is our dataset (SVM works with order O(n^2) to O(n^3))?
# Let's take a look to decide whether we must downsample the dataset

print(len(df)) # almost 30000, very big
print(len(df[df["default payment next month"] == 0])) # almost 4 times as many zeros as ones. Not extreme, but
                                                      # the downsample could be helpful

df_ones = df[df["default payment next month"] == 1]
df_zeros = df[df["default payment next month"] == 0]

df_ones = resample(df_ones, n_samples = 1000, replace = False, random_state = 42)
df_zeros = resample(df_zeros, n_samples = 1000, replace = False, random_state = 42)
df = pd.concat([df_ones, df_zeros])

# Lastly, we have to turn the previous variables (and "SEX") into dummy variables to avoid the finding of relations
# that don't exist in the model (for example, it might relate 1 and 2 in "EDUCATION" closer than 1 and 4, but they are
# a way to  discretely categorise the variable in graduate school, university and others, so there isn't a sense of
# distance as in continuous variables
X = df.iloc[:,1:-1] # We also drop the ID column since it is useless for the training
y = df.iloc[:,-1]
X = pd.get_dummies(X, columns = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
                   dtype = "int", drop_first = True) # Last argument because one of the categories is always completely
                                                     # characterized by the others

# It is a good decision to use a Support Vector Machine, since the space dimension is high (after the dummy variables
# the number of columns is 89)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3,
                                                   shuffle = True, stratify = y)
X_train, X_test = scale(X_train), scale(X_test)
y_train, y_test = y_train.astype(int).values, y_test.astype(int).values   # It must be specified that 0 and 1 are
                                                                          # INTEGERS, if not it can lead to errors
print(y_train)

svc = SVC(random_state = 42, gamma = "auto")
svc.fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(
    svc,
    X_test,
    y_test,
    display_labels = ["Not default", "Default"]
)
plt.show()

# We can still upgrade this result by searching through a GridSearchCV and CrossValidation, trying to find the best
# values for gamma and C (parameters of svm):
# C = Regularization parameter -> Low C encourages larger margin, more misclassifications but less
#                                 overfitting/generalization, while high C does the opposite

# gamma = Kernel Width -> Low gamma means the radius of influence of the samples selected by the model is bigger.
#                         That is, the decision boundaries are simpler, since a lot of samples are accepted, which
#                         can lead to underfitting. high gamma means the opposite, so it can lead to overfitting


param_grid = {"C": [0.1, 0.5, 1], "gamma": [0.1, 0.01, 0.001], "kernel": ["rbf", "poly", "sigmoid"]}

optimal_param_search = GridSearchCV(
    SVC(),
    param_grid,
    cv = 5,
    scoring = "accuracy",
    verbose = 0
)
optimal_param_search.fit(X_train, y_train)
print(optimal_param_search.best_params_)

# Now we use this new values for the parameters
svc = SVC(random_state = 42, gamma = 0.01, C = 0.5, kernel = "rbf")
svc.fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(
    svc,
    X_test,
    y_test,
    display_labels = ["Not default", "Default"]
)
plt.show()

# We can obtain the values for describing the fit: support vectors and dual coefs, attending to the following formula:
# f(x) = sum_{i in SV}(alpha_i * y_i * K(x_i, x)) + b = 0
# where alpha_i are the lagrange multipliers, y_i are the true class label (the real values of the sample, turned into
# +1 and -1) and K is the kernel function
sup_vect = svc.support_vectors_
dual_coefs = svc.dual_coef_[0]   # (alpha * y)_i
intercept = svc.intercept_[0]

with np.printoptions(threshold = np.inf):       # with allows to set options for the lines in it, and return to the
                                                # original settings after the lines are finished
    print("Obtained support vectors: ", sup_vect)
    print(sup_vect.shape)
    print("Dual coefs for the non-linear (rbf) function: ", dual_coefs)
    print(len(dual_coefs))
    print("Intercept: ", intercept)

# The result has not improved too much overall, better for not default (0) but worse for default (1)
#
# We can try to plot the data, using what's called a PCA (Principal Component Analysis), basically finding
# the (given to the function) number of vectors that best describes the data being orthogonal to the others.
# Nonetheless, first we must check that there are 2 dimensions which stand out considerably more than the others
pca = PCA()

X_pca = pca.fit_transform(df.iloc[:,1:-1])

per_var = np.round(pca.explained_variance_ratio_*100, decimals = 0)
print(sum(per_var[0:2]))
plt.bar(x = range(1, len(per_var)+1), height = per_var)
plt.ylabel("variances")
plt.xlabel("vectors")
plt.show()

# There is a great difference between the first 2 eigenvectors and the others, so it looks like the representation
# is valid for the original sample. However, the training was made with the 80 columns sample, so we have to focus
# on the latter
X_pca = pca.fit_transform(X_train)

per_var = np.round(pca.explained_variance_ratio_*100, decimals = 0)
print(sum(per_var[0:2]))
plt.bar(x = range(1, len(per_var)+1), height = per_var)
plt.ylabel("variances")
plt.xlabel("vectors")
plt.show()

# The result is not good. The first 2 biggest variances sum up to 18%, so it is far from representing the whole
# sample. We conclude in this case it is difficult to make a good 2D representation. We could consider using MCA
# to apply directly to categorical and numerical data, or t-SNE and UMAP (no idea about them yet hahah)

##############
### DAY 11 ###
##############

## Today's focus is on learning the specific applications of the kernel functions. We are going to start with
## a designed kernel function (the one in Day 12 infograph from the 100 Days Challenge) which is the sum of
## the squared components in a 2D continuous features dataset. For that, we will use the make_circles function
## from sklearn and carry out the necessary adjustments

Xsample, ysample = make_gaussian_quantiles(n_samples = 100, n_features = 2, n_classes = 2, shuffle = True)
index_adm = np.where(ysample == 1)
index_nadm = np.where(ysample == 0)
plt.scatter(Xsample[index_adm, 0], Xsample[index_adm, 1], edgecolors = "k")
plt.scatter(Xsample[index_nadm, 0], Xsample[index_nadm, 1], edgecolors = "k")
plt.show()

# y is not the class label, but the 2nd dim of X, and the SVC only uses X information for its calculations
# and keepdims reassures that the dimension (in this case 2) is the same, being (350,1) instead of (350,2)

def kernel_radius(x,y):
    r_X = np.sum(x**2, axis = 1, keepdims = True)
    r_Y = np.sum(y**2, axis = 1, keepdims = True)
    return np.dot(r_X, r_Y.T)

svc = SVC(kernel = kernel_radius)
X_train, X_test, y_train, y_test = train_test_split(Xsample, ysample, random_state = 42, test_size = 0.3,
                                                   shuffle = True, stratify = ysample)

svc.fit(X_train, y_train)
y_test = y_test.astype(int).reshape(-1,1)
y_pred = svc.predict(X_test)

ConfusionMatrixDisplay.from_estimator(
    svc,
    X_test,
    y_test,
    display_labels = ["Clase 1", "Clase 2"]
)
plt.show()

## This is, however, very computationally expensive, O(N^2) or above. that's why we can define a variable
## r2 = x**2 + y**2 and use it with a linear kernel, O(N), as followed (LinearKernel would also work)

X_r2 = np.sum(Xsample**2, axis = 1, keepdims = True)

X_r2 = np.sum(Xsample**2, axis = 1, keepdims = True) # we define it again to consider the actual time it
                                                     # would take without the axes plot

svc = SVC(kernel = "linear")
X_train, X_test, y_train, y_test = train_test_split(X_r2, ysample, random_state = 42, test_size = 0.3,
                                                   shuffle = True, stratify = ysample)
svc.fit(X_train, y_train)
print(svc.coef_)
print(svc.intercept_)
r2_plane = -svc.intercept_[0]/svc.coef_[0,0]

fig, axes = plt.subplots(nrows = 1, ncols = 2)
axes[0].scatter(Xsample[index_adm, 0], X_r2[index_adm], edgecolors = "k")
axes[0].scatter(Xsample[index_nadm, 0], X_r2[index_nadm], edgecolors = "k")
axes[1].scatter(Xsample[index_adm, 1], X_r2[index_adm], edgecolors = "k")
axes[1].scatter(Xsample[index_nadm, 1], X_r2[index_nadm], edgecolors = "k")
axes[0].axhline(y = r2_plane, color = "r")
axes[1].axhline(y = r2_plane, color = "r")
axes[0].set_title(r"x vs r^2")
axes[1].set_title(r"y vs r^2")
plt.show()

# We find the values of the grid that form the circle with found radius
theta = np.linspace(0,2*np.pi, 501)
r_plane = np.sqrt(r2_plane)
x1, x2 = r_plane*np.cos(theta), r_plane*np.sin(theta)
plt.scatter(Xsample[index_adm, 0], Xsample[index_adm, 1], edgecolors = "k")
plt.scatter(Xsample[index_nadm, 0], Xsample[index_nadm, 1], edgecolors = "k")
plt.plot(x1, x2, c = "r")
plt.show()

ConfusionMatrixDisplay.from_estimator(
    svc,
    X_test,
    y_test,
    display_labels = ["Clase 1", "Clase 2"]
)
plt.show()

## Somehow it seems like the defined kernel is a bit better on this occasion, but just because
## the time taken is mostly due to other python operations such as arrays transformations, shuffling
## the data or implementing the different functions. As soon as the number of samples grows, the
## time to give the wanted outcome becomes substantially lower with the linear kernel (for example,
## with 20000 samples the time of the defined kernel goes up to 1.3s, but the linear one gives 0.24s)
