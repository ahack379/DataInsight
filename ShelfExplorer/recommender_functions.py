from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import precision_score


# Now fill the rank matrix and validation matrix
def build_rank_matrix(df_u,df_b):
    """
    : param df_u : pandas df that has book rating info per user
    : param df_b : pandas df that has book rating info per book

    returns:
      rating matrix, genre matrix

    These matrices are the same size. The same entries in the matrix are 
    filled with either ratings or genre tags for convenient use later
    """

    N_BOOKS = len(df_u.book_id.unique()) 
    N_USERS = len(df_u.user_id.unique())

    # First, define (n_books x n_users) matrices to store ratings and genre tags
    ratings_mat = np.zeros((N_USERS,N_BOOKS))
    genre_mat = np.zeros((N_USERS,N_BOOKS))

    # Now fill the rank matrix and validation matrix
    for i in range(df_u.shape[0]):
        user_i = df_u.user_idx.values[i] # This goes from 0 -> 536
        book_i = df_u.book_idx.values[i] # This goes from 0 -> 7336
        rating_i = df_u.rating.values[i] # This goes from 1 -> 5
    
        # Fill ratings matrix
        ratings_mat[user_i][book_i] = rating_i
        
        # Now fill the genre tag matrix
        # First link ratings matrix entry to its book in df_b_ratings
        actual_book_i = df_u.book_id.values[i] # book_id from 1 -> 10000
        book_q = df_b.loc[df_b['book_id'] == actual_book_i]
        tag_id = book_q.tag_id.values[0] 
        
        genre_mat[user_i][book_i] = int(tag_id)    

    return ratings_mat, genre_mat

def impute_matrix(ratings_mat):

    """
    : param ratings_mat :

    returns:
      imputed ratings matrix

    This function takes in the ratings matrix we just built
    and fills in each zero with an average of user and book ratings
    at that location
     
    """
    user_ave_v = ratings_mat.sum(1)/(ratings_mat!=0).sum(1).astype(float)
    book_ave_v = ratings_mat.T.sum(1)/(ratings_mat.T!=0).sum(1).astype(float)
    
    n_users = ratings_mat.shape[0]
    n_books = ratings_mat.shape[1]
    
    ave_mat = np.zeros((n_users,n_books))
    user_ave_mat = np.zeros((n_users,n_books))
    
    for i in range(n_users):
        rowi = [ (user_ave_v[i]+book_ave_v[j])/2 for j in range(n_books) ]
        ave_mat[i] = rowi
        
        row_ave = [user_ave_v[i] for j in range(n_books)]
        user_ave_mat[i] = row_ave

    # Per empty entry, set entry value to average of user and book rankings
    ratings_mat[ratings_mat == 0] = ave_mat[ratings_mat == 0]

    return ratings_mat, user_ave_mat


def get_user_genre_pref(df_tags,genre_mat):

    """

    : params df_tags : pandas data frame of top genre tag ids and names
    : params genre_mat : matrix size users x books of genre tag id
    
    returns :
      top genre per user
    
    Get the top genre per user to mask out later 

    """
    n_users = genre_mat.shape[0]
 
    # Now store the top genre preferences per user 
    top_genres_per_user = []
    
    for rowi in range(genre_mat.shape[0]):
        
        i = genre_mat[rowi]
        unique, counts = np.unique(i,return_counts=True)
    
        pred_idxs_sorted = np.argsort(counts)
        pred_idxs_sorted = pred_idxs_sorted[::-1]
    
        df_fav = df_tags.query('tag_id == %d'%int(unique[pred_idxs_sorted[1]]))
        top_genres_per_user.append(df_fav['tag_id'].values[0])
            
    return top_genres_per_user

def blind_users_fav_genre(genre_mat,top_genres):

    """
    : params genre_mat : matrix size users x books of genre tag id
    : params top_genres : list of favorite genres per user
    
    returns :
      matrix blinded from users favorite genre
    
    Blind the users favorite genre from the final recommendations

    """

    # First identify all locations with ratings data
    blind_mat = np.where(genre_mat > 0, 1, 0)
    
    # Next, blind out elements where the genre is the users favorite
    for i in range(genre_mat.shape[0]):
        rowi = genre_mat[i]
        blind_mat[i] = [ 1 if genre_mat[i][j] != 0 and genre_mat[i][j] != top_genres[i]\
                            else 0 for j in range(genre_mat.shape[1])]
    
    return blind_mat

def train_test_split(ratings,split=30):
    """
    : params ratings : ratings matrix 
    : params split : number of entries to include in the test set     
 
    Split ratings matrix into train and test set
    """
    
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=split, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# Look at top k suggestions; k=30 is the mse minimum calcualted elsewhere
def predict_topk(ratings, similarity, kind='user', k=30):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]

            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))

    if kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
    
    return pred

def binarize(arr, tol):
    """
    Convert continous valued array to binary. 
    """
    arr[arr < tol] = 0
    arr[arr >= tol] = 1
    return arr

def precision_at_k(true, pred, pred_binarized, user_ids, k, tol=[]):
    unique_users = np.unique(user_ids)
    precisions = np.zeros(unique_users.size)
    
    for i in range(unique_users.size):
        user_ind = user_ids == unique_users[i]
        user_true = true[user_ind]
        user_pred = pred[user_ind]
        user_pred_binarized = pred_binarized[user_ind]
        ranked_ind = np.argsort(-user_pred)[:k]
        precisions[i] = precision_score(user_true[ranked_ind], user_pred_binarized[ranked_ind])
    return np.mean(precisions[precisions > 0]) #precisions
