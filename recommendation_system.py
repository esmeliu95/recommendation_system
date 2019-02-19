
# coding: utf-8

# In[4]:


import pandas 
from datetime import datetime
from surprise import SVD
from surprise import Reader 
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
import numpy as np
from collections import defaultdict

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

    return trainset, testset, gender, release_year


if __name__ == "__main__":

    # prepare data
    print('Extracting data from the ml-100k dataset ...')

    
    trainset, testset, gender, release_year = prepare_movielens_data(data_path='../ml-100k/')
    
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


data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = SVD(n_factors=7, reg_all=0.1)
algo.fit(trainset)
testset = trainset.build_testset()

predictions = algo.test(testset)
print('task 1')
mae = accuracy.mae(predictions)
print('accuracy: ',mae) #task 1


# In[5]:


def get_top_n(predictions, n=5):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First train an SVD algorithm on the movielens dataset.
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = SVD(n_factors=6, reg_all=0.1)
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()

predictions = algo.test(testset)
top_n = get_top_n(predictions, n=5)

# Print the recommended items for each user
res = []
for uid, user_ratings in top_n.items():
    res.append((uid, [iid for (iid, _) in user_ratings]))


# In[6]:


temp_test = pandas.read_csv('testset.csv',header=None)
test = temp_test.values[1:]
table = np.zeros((len(res),5))
c_test = defaultdict(list)
for j in test:
  
    sub = {}
    sub[j[1]] = j[2]    
    c_test[j[0]].append(sub)

#ctest: {user: list[{movie1: rating},{movie2: rating}]}
for big_i, i in enumerate(res):
    if i[0] in c_test: 
        for j in c_test[i[0]]: #list[{movie1: rating},{movie2: rating}]
            for index, k in enumerate(i[1]):
            #for k in i[1]: #list[movie1, movie2, ...]
                if k in j:
                    table[big_i][index] = int(j[k])

ans = []
for i in range(len(table)):
    for j in range(len(table[i])):
        if table[i][j] == 0:
            table[i][j] = 2

    ans.append(np.mean(table[i]))
    
print('task 2')
print(np.mean(ans))

