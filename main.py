# import libraries
import pandas as pd
# load data
df = pd.read_csv(r"E:\Lectures\Artificial Intelligence\Tumor Cancer Prediction_DataSet.csv")
# drop column with missing value
df.dropna(axis=1)
# drop duplicates

print(df)

# get count of M and B cells
#df['diagnosis'].value_counts()

# look at the data types to see which columns need to be encoded
df.dtypes

# encode the catagorical data vlaues   B = 0, M = 1
from sklearn.preprocessing import LabelEncoder
LabelEncoder_Y = LabelEncoder()
df.iloc[:,31]=LabelEncoder_Y.fit_transform(df.iloc[:,31])
df.iloc[:,31]

#split the data set into independent x and dependant y
X = df.iloc[:,0:31].values   # tell us feature by which could detec cance
Y = df.iloc[:,31].values   # tell us has cancer or not

# split data set int 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25,random_state =0)

#scale the data (feature scaling/ Data Normalization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# create a function for the models
def models(X_train, Y_train):
    ################## SVM Model###########################
    from sklearn import svm
    clf = svm.SVC(kernel='linear')  # Linear Kernel
    clf.fit(X_train, Y_train)

    ################Logistic Regression####################
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    ###############Decision Tree###########################
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    ############## Random forest Classifier################
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)

    # print the models accuracy on the training data
    print('[0]SVM Training Accuracy :', clf.score(X_train, Y_train))
    print('[1]Logisic Regression Training Accuracy :', log.score(X_train, Y_train))
    print('[2]Decision Tree Classifier Training Accuracy :', tree.score(X_train, Y_train))
    print('[3]Random Forest Classifier Training Accuracy :', forest.score(X_train, Y_train))
    return clf, log, tree, forest

# getting all of the models
model = models(X_train, Y_train)

# test model accurecy on tset data on confussion matrix
from sklearn.metrics import confusion_matrix
for i in range (len(model)):
 print('model', i)
 cm = confusion_matrix(Y_test , model[i].predict(X_test))
 TP = cm[0][0]
 TN = cm[1][1]
 FN = cm[1][0]
 FP = cm[0][1]
 print(cm)
 print('testing accuracy :     '+str((TP+TN)/(TP + TN + FN + FP)))
 print(" ")