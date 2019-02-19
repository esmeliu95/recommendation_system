
# coding: utf-8

# In[53]:


import pandas 
from datetime import datetime
from surprise import SVD
from surprise import Reader 
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def prepare_movielens_data(data_path):

    # get user gender, index user ids from 0 to (#user - 1)
    users = pandas.read_csv(data_path + 'u.user', sep='|', header=None, names=['id', 'age', 'gender', 'occupation', 'zip-code'])
    gender = pandas.DataFrame(users['gender'].apply(lambda x: int(x == 'M'))) # convert F/M to 0/1
    user_id = dict(zip(users['id'], range(users.shape[0]))) # mapping user id to linear index

    # the zero-th column is the id, and the second column is the release date
    movies = pandas.read_csv(data_path + 'u.item', sep='|', encoding = 'latin-1', header=None, usecols=[0, 1, 2],
                             names=['item-id', 'title', 'release-year'])

    bad_movie_ids = list(movies['item-id'].loc[movies['release-year'].isnull()]) # get movie ids with a bad release date

    movies = movies[movies['release-year'].notnull()] # item 267 has a bad release year, remove this item
    release_year = pandas.DataFrame(movies['release-year'].apply(lambda x: datetime.strptime(x, '%d-%b-%Y').year))
    movie_id = dict(zip(movies['item-id'], range(movies.shape[0]))) # mapping movie id to linear index

    # get ratings, remove ratings of movies with bad release years.
    rating_triples = pandas.read_csv(data_path + 'u.data', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    rating_triples = rating_triples[['user', 'item', 'rating']] # drop the last column
    rating_triples = rating_triples[~ rating_triples['item'].isin(bad_movie_ids)] # drop movies with bad release years

    # map user and item ids to user indices
    rating_triples['user'] = rating_triples['user'].map(user_id)
    rating_triples['item'] = rating_triples['item'].map(movie_id)

    # the following set assertions guarantees that the user ids are in [0, #users), and item ids are in [0, #items)
    assert(rating_triples['item'].unique().min() == 0)
    assert(rating_triples['item'].unique().max() == movies.shape[0] - 1)
    assert(rating_triples['user'].unique().min() == 0)
    assert(rating_triples['user'].unique().max() == users.shape[0] - 1)
    assert(rating_triples['item'].unique().shape[0] == movies.shape[0])
    assert(rating_triples['user'].unique().shape[0] == users.shape[0])

    # training/test set split
    rating_triples = rating_triples.sample(frac=1, random_state=2018).reset_index(drop=True) # shuffle the data
    train_ratio = 0.9
    train_size = int(train_ratio * rating_triples.shape[0])

    trainset = rating_triples.loc[0:train_size]
    testset = rating_triples.loc[train_size + 1:]

    return trainset, testset, gender, release_year,rating_triples


if __name__ == "__main__":

    # prepare data
    print('Extracting data from the ml-100k dataset ...')

    
    trainset, testset, gender, release_year,all_data = prepare_movielens_data(data_path='../ml-100k/')
    
    trainset.to_csv('trainset.csv', index=False)
    testset.to_csv('testset.csv', index=False)
    gender.to_csv('gender.csv', index=False)
    release_year.to_csv('release-year.csv', index=False)
    
    print('Done')


# In[ ]:


from surprise.model_selection import GridSearchCV
df = pandas.DataFrame(trainset)
reader=Reader(line_format='user item rating',sep=',',skip_lines=1)
data=Dataset.load_from_df(df,reader)
hyper={'n_factors':[5,6,7],'reg_all':[0.1,1,10]}
clf = GridSearchCV(SVD,hyper,cv=5,measures=['mae','rmse'])#rmse
clf.fit(data)
print(clf.best_params)
print(clf.best_score['mae'])


# In[2]:


from collections import defaultdict
from surprise import SVD
from surprise import Dataset


ddf = pandas.DataFrame(all_data)
reader=Reader(line_format='user item rating',sep=',',skip_lines=1)
data=Dataset.load_from_df(ddf,reader)
new_t = data.build_full_trainset()
algo = SVD(n_factors=6, reg_all=0.1)
algo.fit(new_t)

u = algo.pu #user
v = algo.qi #movie feature


# In[17]:

#train gender
gender_data = np.array(gender)
labels = []
for i in gender_data:
    labels.append(i[0])
xtrain,xtest,ytrain,ytest = train_test_split(u, labels, test_size=0.25)


# In[19]:


# Perform 5-fold cross validation
def createRFmodel(xtrain,ytrain,estimators,X,labels):
    clf = RandomForestClassifier(n_estimators=estimators,n_jobs=-1)
    clf.fit(xtrain,ytrain)
    train_clf = RandomForestClassifier(n_estimators=estimators,n_jobs=-1)
    train_clf.fit(X,labels)
    acc = train_clf.score(X,labels)
    return clf,acc


est = [1,10,50,100,800]
validation_list = []
train_list = []
for i in est:
    new_model,accuracy = createRFmodel(xtrain,ytrain,i,u,labels)
    scores = cross_val_score(new_model, u, labels, cv=5)
    validation_list.append(np.mean(scores))#(1-(np.mean(scores)))
    t_scores = accuracy#1-accuracy
    train_list.append(t_scores)    


print('validation accuracy are ',validation_list)
print('trainning accuracy are ',train_list)


# In[52]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

year_data = np.array(release_year)
labels_year = []
for i in year_data:
    labels_year.append(i[0])

#split into training and testing
xtrain_y,xtest_y,ytrain_y,ytest_y = train_test_split(v, labels_year, test_size=0.25)
clf = LinearRegression().fit(xtrain_y, ytrain_y)
y_pred  = clf.predict(xtest_y)
regression = mean_squared_error(ytest_y, y_pred)
print('regression mse: ',regression)#regression mse


# In[51]:

#calculate naive mase
naive_y = np.mean(ytrain_y)

naive_y_test = []
for i in range(len(ytest_y)):
    naive_y_test.append(int(naive_y))

naive = mean_squared_error(naive_y_test,ytest_y)
print('naive mse: ',naive)#naive mse

