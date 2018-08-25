import csv
import pprint
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('compete.csv', header=0)
data = data.dropna()
print(list(data.columns))

print(data.shape)
print(list(data.columns))
print(data.head())
print(data['Speaker'].unique())
print(data['Token'].value_counts())
sns.countplot(x='Subj case',data=data,palette='hls')
plt.show()
print(data.groupby('Subj case').mean())

# TO normalize: divide dative bar by total datives, and nom bar by total noms (normalized frequency)
plt.style.use('grayscale')
#Plot subject case by register:
pd.crosstab(data['Register'],data['Subj case'],normalize='columns').plot(kind='bar')
plt.title('Subject case by register')
plt.xlabel('Register', size='medium', weight='bold')
plt.ylabel('Normalised frequency', size='medium', weight='bold')
plt.xticks(np.arange(5), ['Least formal','Less formal','Neutral','More formal','Most formal'], rotation=0)
plt.savefig('subjcasebyreg')
plt.gray()
plt.show()
#Plot subject case by lexeme:
pd.crosstab(data['Lexeme'],data['Subj case'],normalize='columns').plot(kind='bar')
plt.title('Subject case by lexeme')
plt.xlabel('Lexeme', size='medium', weight='bold')
plt.ylabel('Normalised frequency', size='medium', weight='bold')
plt.xticks(np.arange(2), ['Danish','Old Norse'], rotation=0)
plt.savefig('subjcasebylex')
plt.show()
#Plot subject case by speaker age:
pd.crosstab(data['Speaker age'],data['Subj case'],normalize='columns').plot(kind='bar')
plt.title('Subject case by speaker age')
plt.xlabel('Speaker age', size='medium', weight='bold')
plt.ylabel('Normalised frequency', size='medium', weight='bold')
plt.xticks(np.arange(5), ['Youngest','Younger','Middle','Older','Oldest'], rotation=0)
plt.savefig('subjcasebyspeakerage')
plt.show()

clusters_train = data['Speaker']
print(clusters_train)

data.drop(data.columns[[0, 1, 6, 7]], axis=1, inplace=True)

data2 = pd.get_dummies(data, columns =['Register', 'Speaker age', 'Lexeme'])

X = data2.iloc[:,1:]
y = data2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

sns.heatmap(data2.corr(), cmap="YlGnBu")
plt.show()

classifier = LogisticRegression(random_state=0)
print(classifier.fit(X_train, y_train))

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

Z_train = np.ones((271,1))
print(Z_train)
from merf import MERF
mrf = MERF(n_estimators=300, max_iterations=100)
mrf.fit(X_train, Z_train, clusters_train, y_train)

cat_vars = ['Speaker', 'Token', 'Item']
for var in cat_vars:
	cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(data[var], prefix=var)
	data1 = data.join(cat_list)
	data = data1

data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]

data_final = data[to_keep]
data_final_vars = data_final.columns.values.tolist()
y = ['Subj case']
X = [i for i in data_final_vars if i not in y]
print(data_final_vars)

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 18)
rfe = rfe.fit(data_final[X], data_final[y].values.ravel())
print(rfe.support_)
print(rfe.ranking_)

cols = ['Speaker age', 'Lexeme', 'Speaker_asajoh', 'Speaker_birkblog', 'Speaker_fridasvein', 'Speaker_hermansstova', 'Speaker_kinnapoulsen', 'Speaker_listinblog', 'Speaker_okkara-didda', 'Speaker_roskur', 'Speaker_samalsdiary', 'Speaker_winthereig', 'Token_daamar', 'Token_daami', 'Token_mangli', 'Token_toervar', 'Item_lack', 'Item_like', 'Item_need']
X = data_final[cols]
y = data_final['Subj case']

import statsmodels.api as sm
logit_model = sm.Logit(y_test,X_test)
result = logit_model.fit()
print(result.summary())

