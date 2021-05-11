import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

if __name__ == "__main__":
    trainSet = pd.read_csv('train.csv')
    testingSet = pd.read_csv('test.csv')
    survived = trainSet[trainSet['Survived'] == 1]
    not_survived = trainSet[trainSet['Survived'] == 0]

    survival_length = len(survived)
    divisionCalc = len(survived)/len(trainSet)
    survival_percentage = float(divisionCalc)*100.0
    not_survival_length = len(not_survived)
    divisionCalc = len(not_survived)/len(trainSet)
    not_survival_percentage = float(divisionCalc)*100.0

    trainSet.Pclass.value_counts()
    trainSet.groupby('Pclass').Survived.value_counts()
    grouping = trainSet[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False)
    grouping.mean()

    trainSet.Sex.value_counts()
    trainSet.groupby('Sex').Survived.value_counts()
    grouping = trainSet[['Sex', 'Survived']].groupby(['Sex'], as_index=False)
    grouping.mean()

    tab = pd.crosstab(trainSet['Pclass'], trainSet['Sex'])

    tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
    plt.xlabel('Pclass')
    plt.ylabel('Percentage')
    #plt.show()

    trainSet.Embarked.value_counts()
    surviveCalc = trainSet.groupby('Embarked').Survived
    surviveCalc.value_counts()
    grouping = trainSet[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False)
    grouping.mean()

    trainSet.Parch.value_counts()
    surviveCalc = trainSet.groupby('Parch').Survived
    surviveCalc.value_counts()
    grouping = trainSet[['Parch', 'Survived']].groupby(['Parch'], as_index=False)
    grouping.mean()

    trainSet.SibSp.value_counts()
    surviveCalc = trainSet.groupby('SibSp').Survived
    surviveCalc.value_counts()
    grouping = trainSet[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False)
    grouping.mean()

    trainSet_testingSet_data = [trainSet, testingSet]
    for data in trainSet_testingSet_data:
        name = data.Name
        stringify = name.str
        data['Title'] = stringify.extract(' ([A-Za-z]+)\.')

    pd.crosstab(trainSet['Title'], trainSet['Sex'])

    for data in trainSet_testingSet_data:
        data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
        #------Replacements-------
        data['Title'] = data['Title'].replace('Mlle', 'Miss')
        data['Title'] = data['Title'].replace('Ms', 'Miss')
        data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    grouping = trainSet[['Title', 'Survived']].groupby(['Title'], as_index=False)
    grouping.mean()

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
    for data in trainSet_testingSet_data:
        #-----Mapping
        data['Title'] = data['Title'].map(title_mapping)
        #-----NA fill-----
        data['Title'] = data['Title'].fillna(0)

    for data in trainSet_testingSet_data:
        data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    trainSet.Embarked.unique()
    trainSet.Embarked.value_counts()

    for data in trainSet_testingSet_data:
        #-----fill NA------
        data['Embarked'] = data['Embarked'].fillna('S')

    for data in trainSet_testingSet_data:
        #------Map it------
        mapIt = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
        data['Embarked'] = mapIt.astype(int)

    for data in trainSet_testingSet_data:
        #-----mean-----
        age_avg = data['Age'].mean()
        #-----std-----
        age_std = data['Age'].std()
        #-----isNull and sum-----
        age_null_count = data['Age'].isnull().sum()
    
        ageSub = age_avg - age_std
        ageAdd = age_avg + age_std
        age_null_random_list = np.random.randint(ageSub, ageAdd, size=age_null_count)
        data['Age'][np.isnan(data['Age'])] = age_null_random_list
        data['Age'] = data['Age'].astype(int)
    
    trainSet['AgeBand'] = pd.cut(trainSet['Age'], 5)

    #print(trainSet[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

    for data in trainSet_testingSet_data:
        data.loc[ data['Age'] <= 16, 'Age'] = 0
        data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
        data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
        data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
        data.loc[ data['Age'] > 64, 'Age'] = 4

    for data in trainSet_testingSet_data:
        data['Fare'] = data['Fare'].fillna(trainSet['Fare'].median())

    trainSet['FareBand'] = pd.qcut(trainSet['Fare'], 4)
    #print (trainSet[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
    for data in trainSet_testingSet_data:
        data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
        data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
        data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
        data.loc[ data['Fare'] > 31, 'Fare'] = 3
        data['Fare'] = data['Fare'].astype(int)

    for data in trainSet_testingSet_data:
        data['FamilySize'] = data['SibSp'] +  data['Parch'] + 1

    #print (trainSet[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

    for data in trainSet_testingSet_data:
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
    
    #print (trainSet[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

    features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
    trainSet = trainSet.drop(features_drop, axis=1)
    testingSet = testingSet.drop(features_drop, axis=1)
    trainSet = trainSet.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)
    #print(trainSet.head())

    X_trainSet = trainSet.drop('Survived', axis=1)
    y_trainSet = trainSet['Survived']
    X_testingSet = testingSet.drop("PassengerId", axis=1).copy()

    X_trainSet.shape, y_trainSet.shape, X_testingSet.shape

    #-------Logistic Regression-----
    clf = LogisticRegression()
    clf.fit(X_trainSet, y_trainSet)
    y_pred_log_reg = clf.predict(X_testingSet)
    acc_log_reg = round( clf.score(X_trainSet, y_trainSet) * 100, 2)
    print ("Using Logistic Regression:", str(acc_log_reg) + ' percent')

    #--------Stochastic gradient descent------
    clf = SGDClassifier(max_iter=5, tol=None)
    clf.fit(X_trainSet, y_trainSet)
    y_pred_sgd = clf.predict(X_testingSet)
    acc_sgd = round(clf.score(X_trainSet, y_trainSet) * 100, 2)
    print ("Using Stochastic Gradient Descent:", acc_sgd, 'percent')

    submission = pd.DataFrame({
        "PassengerId": testingSet["PassengerId"],
        "Survived": y_pred_log_reg
    })
    #convert into submission
    submission.to_csv('submission.csv', index=False)
    

