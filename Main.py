import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
#%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pandas as pd
import warnings
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


#import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



# Čitanje CSV datoteke u DataFrame
df = pd.read_csv('weatherAUS.csv')

# Ispisivanje prvih nekoliko redova DataFrame-a
print(df.head())

warnings.filterwarnings('ignore')
print(df.isnull().any())#gledamo da li u nekim vrstama i kolonama ima null vrednosti
col_names = df.columns
print(col_names)
#df.drop(['RISK_MM'], axis=1, inplace=True)
df.info()

categorical = [var for var in df.columns if df[var].dtype=='O']

print('Imamo {} varijable u kategorijama\n'.format(len(categorical)))

print('Kategorijske varijable su :', categorical)
print(df[categorical].head())
print(df[categorical].isnull().sum())#proveravamo nedostajuce vrednosti u kategorickim varijablama
cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())#ispisuje sve kolone koje imaju null-ove

for var in categorical:
    print(df[var].value_counts())#proveravanje ucestalosti kategorijskih varijabli

for var in categorical:
    print(var, ' contains ', len(df[var].unique()), ' labels')#proveravamo kardinalitete u kategorickim varijablama

df['Date'].dtypes
df['Date'] = pd.to_datetime(df['Date']) #analizira datume,trenutno kodirane kao stringove,u format datuma i vremena

df['Year'] = df['Date'].dt.year #odvaja godinu od datuma
df['Year'].head()

df['Month'] = df['Date'].dt.month #odvaja mesec od datuma
df['Month'].head()

df['Day'] = df['Date'].dt.day #odvaja dan od datuma
df['Day'].head()
df.info() #ponovo gledamo rezime skupa podataka

df.drop('Date', axis=1, inplace = True) #dropujemo originalnu promenljivu 'Date'
df.head() #ponovo pregledajmo skup podataka

categorical = [var for var in df.columns if df[var].dtype=='O'] #trazimo kategoricke varijable

print('There are {} categorical variables\n'.format(len(categorical)))     #primecujemo da sada ima 6 kategorija
print(df[categorical])                                                     #jer smo dropovali kolonu 'Date'

print('The categorical variables are :', categorical)
df[categorical].isnull().sum()      #trazimo nedostajuce vrednosti
print('Location contains', len(df.Location.unique()), 'labels')  #printuj broj labela u 'Location' varijabli
print(df.Location.unique())        #ispisuje sve jedinstvene lokacije
print(df.Location.value_counts()) #prikazuje koliko se puta pojavljuje svako ime lokacije

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
df['WindGustDir'].unique()
df.WindGustDir.value_counts()

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
df['WindDir9am'].unique()
df['WindDir9am'].value_counts()

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
df['WindDir3pm'].unique()
df['WindDir3pm'].value_counts()

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
df['RainToday'].unique()
df.RainToday.value_counts()

numerical = [var for var in df.columns if df[var].dtype!='O']       #pronalazimo numericke vrednosti

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
print(df[numerical].head())        #pregled  numerickih vrednosti
df[numerical].isnull().sum()        #proveravamo nedostajuce vrednosti kod numerickih vrednosti
print(round(df[numerical].describe()),2) #pregled zbirne statistike u numerickim varijablama

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')



IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)       #pronalazimo izuzetke u promenljivoj 'Rainfall'
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)     #pronalazimo izuzetke u promenljivoj 'Evaporation'
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)   #pronalazimo izuzetke u promenljivoj 'WindSpeed9am'
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)      #pronalazimo izuzetke u promenljivoj 'WindSpeed3pm'
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

df = df.dropna()
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train.shape, X_test.shape     #proveravanje dimenzija
X_train.dtypes      #proveravamo tip podatka u X_train

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']    #prikaz kategorickih varijabla
print('k.varijable su:',categorical)

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']      #prikaz numerickih varijabla
print('n.varijable su:',numerical)

X_train[numerical].isnull().sum() #proveravamo nedostajuce vrednosti u  numerickim varijablama u X-trainu
X_test[numerical].isnull().sum()   #proveravamo nedostajuce vrednosti u  numerickim varijablama u X-testu
print('#################################################################################')
for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4)) #stampamo procenat od nedostajucih vrednosti u numerickoj varijabli u training set-u

for df1 in [X_train, X_test]:       #ubaciti nedostajuce vrednosti u X_train i
    for col in numerical:           #X_test sa odgovarajucim medijanom kolone u X_trainu
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)
X_train[numerical].isnull().sum()
X_test[numerical].isnull().sum()
X_train[categorical].isnull().mean() #stampa procenat od nedostajucih
                                    # vrednosti u kategoricnim varijablama u training set-u

for col in categorical:     #stampa kategoricke varijable sa nedostajucim vrednostima
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)       #ubaciti nedostajuce kategoricke promenljive sa najcescom vrednoscu
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

X_train.Rainfall.max(), X_test.Rainfall.max()
X_train.Evaporation.max(), X_test.Evaporation.max()
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
print(X_train[numerical].describe())

#categorical
X_train[categorical].head()

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

X_train.head()
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location),
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
X_train.head()

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location),
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
X_test.head()

cols = X_train.columns
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.describe()

# train a logistic regression model on the training set
X_train = X_train.drop(X_train.columns[1], axis=1)
X_test = X_test.drop(X_test.columns[1], axis=1)
'''for col in numerical:
    col_median = X_train[col].median()
    X_train[col].fillna(col_median, inplace=True)
    X_test[col].fillna(col_median, inplace=True)'''
print(categorical)
categorical.remove('Location')
# Handle missing values in categorical features
'''for col in categorical:
    col_mode = X_train[col].mode()[0]
    X_train[col].fillna(col_mode, inplace=True)
    X_test[col].fillna(col_mode, inplace=True)'''
print(X_train)

print(len(X_train),len(y_train))
# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)

y_pred_test = logreg.predict(X_test)

logreg.predict_proba(X_test)[:,0]
logreg.predict_proba(X_test)[:,1]

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
y_pred_train = logreg.predict(X_train)
print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)
logreg100.fit(X_train, y_train)
print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)
logreg001.fit(X_train, y_train)
print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
y_test.value_counts()
null_accuracy = (22067/(22067+6372))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

# Instantiate the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# Fit the Random Forest classifier to the training data
rf_classifier.fit(X_train, y_train)

# Predict the target variable using the trained Random Forest classifier
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the performance of the Random Forest classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print('Random Forest classifier accuracy score: {0:0.4f}'.format(accuracy_rf))



