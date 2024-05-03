
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
# from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import ensemble

from sklearn.svm import SVC
import classification
data, mech = classification.get_data()
cols = ['redshift', 'SFR_onset_frac', 'logM', 'C_SF', 'R_SF', 'Rinner', 'Router']
X_train, X_test, y_train, y_test = train_test_split(data, mech, test_size=0.2)
# use random forest classifier to see which parameter is most important
# clf = ensemble.RandomForestClassifier(n_estimators=20)
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))
# importances = np.full((100, 7), -1.0)
# for i in range(100) :
#     importances[i] = clf.feature_importances_
# importances = np.median(importances, axis=0)
# print(importances)

# classifier = SVC(kernel='linear').fit(X_train, X_test)

Nmeshgrid = 15
lo, hi = np.percentile(X_train, [10, 90], axis=0)
# loop through all the combinations of the parameters
for i in range(len(cols)) :
    for j in range(i, len(cols)) :
        
        # create NxN meshgrid that samples parameter space
        xvals = np.linspace(lo[i], hi[i], Nmeshgrid)
        yvals = np.linspace(lo[j], hi[j], Nmeshgrid)
        XX, YY = np.meshgrid(xvals, yvals)



dataset = datasets.load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(dataset['data'],
    dataset['target'], test_size=0.2, random_state=0)

# clf = tree.DecisionTreeClassifier(class_weight='balanced', random_state=0)
clf = ensemble.RandomForestClassifier(class_weight='balanced', n_estimators=20,
                                      random_state=0)
clf.fit(X_train.values, y_train)
# print(clf.score(X_test.values, y_test))

cols = X_train.columns
sortids = np.argsort(clf.feature_importances_)[::-1]

qs = X_train.quantile(q=[0.1, 0.9])
means = X_train.mean()
stds = X_train.std()

n_sample = 1000
N_meshgrid = 15

cols_plot = cols[sortids[:3]]
ncols = len(cols)

for j in range(3) :
    for i in range(j+1, 3) :
        
        # Get the two column names
        coly = cols_plot[j]
        colx = cols_plot[i]
        
        # Create an NxN meshgrid spanning roughly the range in the data 
        # (from 10th to 90th quartile to exclude obvious outlier)
        yvals = np.linspace(qs.loc[0.1, coly], qs.loc[0.9, coly], N_meshgrid)
        xvals = np.linspace(qs.loc[0.1, colx], qs.loc[0.9, colx], N_meshgrid)
        Xarr, Yarr = np.meshgrid(xvals, yvals)
        
        # Unravel the meshgrid arrays so we have N^2 pairs of (x,y) values
        Xpoints, Ypoints = np.vstack([Xarr.ravel(), Yarr.ravel()])
        
        # Now we have N^2 values of the two features we want to plot.
        # We still need to populate the other features with medians of the dataframe.
        
        # Create an array of random values for our fake data
        fake_data = np.random.normal(loc=means.values, scale=stds.values,
            size=(N_meshgrid*N_meshgrid*n_sample, ncols))
        fake_data = fake_data.reshape((N_meshgrid*N_meshgrid, n_sample, ncols))
        
        # Overwrite the values we want to plot
        fake_data[:, :, sortids[j]] = Ypoints[:, np.newaxis]
        fake_data[:, :, sortids[i]] = Xpoints[:, np.newaxis]
        
        # Expand into N*N*n_sample number of points
        fake_data = fake_data.reshape((N_meshgrid*N_meshgrid*n_sample, ncols))
        
        # Now we can predict the probabilities for our N^2 fake datapoints
        probs = clf.predict_proba(fake_data)
        # Only interested in the positive (1) class
        probs = probs[:, 1]
        
        # Reshape the array back into NxNx5
        probs = np.reshape(probs, (N_meshgrid, N_meshgrid, n_sample))
        
        # Average the probabilities
        probs = np.mean(probs, axis=2)
        
        # Plot the probablity as a function of those two columns
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolormesh(Xarr, Yarr, probs, vmin=0, vmax=1, cmap='RdBu_r')
        ax.set_xlabel(colx)
        ax.set_ylabel(coly)

