from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request, jsonify

#from a_Model import ModelIt

user = 'ah673'
host = 'localhost'

# Import modules for getting the model going
import numpy as np
import math
import random
from scipy import stats
from recommender_functions import *

# Get the ratings by user ID: starts at user index 1 
r = pd.read_csv('../goodbooks-10k/ratings.csv')
df_ratings = pd.DataFrame(r)

## Add counts to the df
cut_u = 175 #175, 1
df_ratings['user_counts'] = df_ratings.groupby(['user_id'])['book_id'].transform('count')
df_ratings['book_counts'] = df_ratings.groupby(['book_id'])['user_id'].transform('count')
df_ratings_cut = df_ratings.query('user_counts > %d'%(cut_u))

df_ratings_cut['user_idx'] = pd.Categorical(df_ratings_cut['user_id']).codes  
df_ratings_cut['book_idx'] = pd.Categorical(df_ratings_cut['book_id']).codes  

# Get the ratings by book 
r = pd.read_csv('../goodbooks-10k/books_with_genres.csv')
df_books = pd.DataFrame(r)
df_books.head()

# Set number of books and users variables for later use
N_BOOKS = len(df_ratings_cut.book_id.unique()) #df_books.shape[0]
N_USERS = len(df_ratings_cut.user_id.unique())
print('N books and userss: ',N_BOOKS, N_USERS)

#ratings_genre_mat = np.zeros((N_USERS,N_BOOKS))
ratings_mat = np.zeros((N_USERS,N_BOOKS))
binary_mat = np.zeros((N_USERS,N_BOOKS))
Y, R = ratings_mat, binary_mat

# Now fill the rank matrix and validation matrix
for i in range(df_ratings_cut.shape[0]):

    user_i = df_ratings_cut.user_idx.values[i] # This goes from 0 -> 536
    book_i = df_ratings_cut.book_idx.values[i] # This goes from 0 -> 7336
    rating_i = df_ratings_cut.rating.values[i] # This goes from 1 -> 5

    ratings_mat[user_i][book_i] = rating_i
    binary_mat[user_i][book_i] = 1 

print('And were in; entering here')

@app.route('/')

@app.route('/index')

def index():
   user = { 'nickname': 'Ariana' } # fake user
   return render_template("index.html",
       title = 'Home',
       user = user)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    my_list = df_books.title.values
    return jsonify(matching_results=my_list)

@app.route('/input')
def user_input():
    my_list = df_books.title.values
    print("\n\n\n\nINPUT List first item: ",my_list[0])
    return render_template("input_new_new.html") #,option_list=my_list)

@app.route('/output')
def user_output():

  book_list = np.zeros(3)
  user_input_list = np.zeros(N_BOOKS)
  top_3_genre_names_per_user = []

  #pull 'book1' from input field and store it
  book_title1 = str(request.args.get('book1'))
  print('OUTPUT TITLE STUFF******************',book_title1)

  new_book1 = df_books.query('title == "%s"' % book_title1)    
  if  new_book1.shape[0] > 0 : 
    book_id = new_book1.book_id.values[0]
    df_temp = df_ratings_cut.query('book_id == %d'% book_id)
    book_idx = df_temp.book_idx.values[0]    
    book_list[0] = int(book_idx)

    genrei = new_book1.genre.values[0]
    if genrei not in top_3_genre_names_per_user:
      top_3_genre_names_per_user.append(genrei)

  book_title2 = str(request.args.get('book2'))
  print('OUTPUT TITLE STUFF******************',book_title2)

  new_book2 = df_books.query('title == "%s"' % book_title2)    
  if  new_book2.shape[0] > 0 : 
    book_id = new_book2.book_id.values[0]
    df_temp = df_ratings_cut.query('book_id == %d'% book_id)
    book_idx = df_temp.book_idx.values[0]    
    book_list[1] = int(book_idx)

    genrei = new_book2.genre.values[0]
    if genrei not in top_3_genre_names_per_user:
      top_3_genre_names_per_user.append(genrei)

  book_title3 = str(request.args.get('book3'))
  print('OUTPUT TITLE STUFF******************',book_title3)

  new_book3 = df_books.query('title == "%s"' % book_title3)    
  if  new_book3.shape[0] > 0 : 
    book_id = new_book3.book_id.values[0]
    df_temp = df_ratings_cut.query('book_id == %d'% book_id)
    book_idx = df_temp.book_idx.values[0]    
    book_list[2] = int(book_idx)

    genrei = new_book3.genre.values[0]
    if genrei not in top_3_genre_names_per_user:
      top_3_genre_names_per_user.append(genrei)

  #if len(top_3_genre_names_per_user) == 0:
  #  my_list = df_books.title.values
  #  return render_template("input_new_new.html",option_list=my_list)

  for i in range(3):
    if int(book_list[i]) == 0:
      continue

    user_input_list[int(book_list[i])] = 5

  print('Checking books; book list: ', book_list)
  print('..and genre list: ', top_3_genre_names_per_user)

  # Make title list for check later
  title_list = []

  for i in book_list:
      df_temp_in = df_ratings_cut.query('book_idx == %d' % i)
      book_id_in = df_temp_in.book_id.values[0]
      new_book_in = df_books.query('book_id == %d' % book_id_in)['title'].values
      title_list.append(new_book_in)

  new_ratings_mat = np.vstack([ratings_mat,user_input_list])

  # Calculate the user similarity and predictions
  user_similarity = fast_similarity(new_ratings_mat,kind='user')
  #user_prediction = predict_fast_simple(new_ratings_mat, user_similarity, kind='user')
  #user_prediction= predict_topk(new_ratings_mat, user_similarity, kind='user', k=30)
  user_prediction = predict_fast_topk(new_ratings_mat, user_similarity, kind='user', k=30)
  
  # Finally, weight the initial predictions made by the system
  weight = 3 
  threshold = 3. / weight
  
  user_prediction_topk_genre = user_prediction.copy()
  user_genre_coords = np.argwhere( user_prediction_topk_genre > threshold)
  
  for c in user_genre_coords:
      x, y = c[0], c[1]
      rating_i = user_prediction_topk_genre[x][y]
  
      # Get book id from the book_idx to link us to the df_books db genre for this rating entry
      ratings_book_id = df_ratings_cut.loc[df_ratings_cut['book_idx'] == int(y)]['book_id'].values[0]
      genre_xy = df_books.loc[df_books['book_id'] == ratings_book_id]['genre'].values[0]
      title_xy = df_books.loc[df_books['book_id'] == ratings_book_id]['title'].values[0]
  
      if genre_xy not in top_3_genre_names_per_user:
          user_prediction_topk_genre[x][y] *= weight
          #print ( 'Weighting this prediction ',genre_xy,title_xy,user_prediction_topk_genre[x][y])
	    

  user_prediction_0 = user_prediction_topk_genre[-1]

  pred_idxs_sorted = np.argsort(user_prediction_0)
  pred_idxs_sorted = pred_idxs_sorted[::-1]

  rec_v = []

  for i in range(20):

      ratings_book_idx = int(pred_idxs_sorted[i])
      df_temp = df_ratings_cut.query('book_idx == %d' % ratings_book_idx)
      book_id = df_temp.book_id.values[0]
      new_book = df_books.query('book_id == %d'%book_id)

      if new_book['title'].values[0] not in title_list: # and new_book['ratings_count'].values[0] < 2000000:

          #print('Predicting rating %0.2f for book "%s", genre: "%s"' % 
          #(user_prediction_0[pred_idxs_sorted[i]],new_book['title'].values[0],new_book['genre'].values[0]))

          url = new_book['image_url'].values[0] 
          book_url_id = "https://www.goodreads.com/book/show/"+str(new_book['goodreads_book_id'].values[0]) 
          title = new_book['title'].values[0] 
          genre_i = new_book['genre'].values[0] 
          author_i = new_book['authors'].values[0] 
          author = author_i.split(',')

          rec_v.append((url,book_url_id,title,author[0],genre_i))
 
  return render_template("output_new.html", recs = rec_v[:3], genres = top_3_genre_names_per_user, the_result = '')

@app.route('/output_genre')
def user_output_genre():

  book_list = np.zeros(3)
  user_input_list = np.zeros(N_BOOKS)
  top_3_genre_names_per_user = []

  #pull 'book1' from input field and store it
  book_title1 = str(request.args.get('book1'))
  print('OUTPUT TITLE STUFF******************',book_title1)
  print('Actualyl in genre out, btw')

  new_book1 = df_books.query('title == "%s"' % book_title1)    
  if  new_book1.shape[0] > 0 : 
    book_id = new_book1.book_id.values[0]
    df_temp = df_ratings_cut.query('book_id == %d'% book_id)
    book_idx = df_temp.book_idx.values[0]    
    book_list[0] = int(book_idx)

    genrei = new_book1.genre.values[0]
    if genrei not in top_3_genre_names_per_user:
      top_3_genre_names_per_user.append(genrei)

  book_title2 = str(request.args.get('book2'))
  print('OUTPUT TITLE STUFF******************',book_title2)

  new_book2 = df_books.query('title == "%s"' % book_title2)    
  if  new_book2.shape[0] > 0 : 
    book_id = new_book2.book_id.values[0]
    df_temp = df_ratings_cut.query('book_id == %d'% book_id)
    book_idx = df_temp.book_idx.values[0]    
    book_list[1] = int(book_idx)

    genrei = new_book2.genre.values[0]
    if genrei not in top_3_genre_names_per_user:
      top_3_genre_names_per_user.append(genrei)

  book_title3 = str(request.args.get('book3'))
  print('OUTPUT TITLE STUFF******************',book_title3)

  new_book3 = df_books.query('title == "%s"' % book_title3)    
  if  new_book3.shape[0] > 0 : 
    book_id = new_book3.book_id.values[0]
    df_temp = df_ratings_cut.query('book_id == %d'% book_id)
    book_idx = df_temp.book_idx.values[0]    
    book_list[2] = int(book_idx)

    genrei = new_book3.genre.values[0]
    if genrei not in top_3_genre_names_per_user:
      top_3_genre_names_per_user.append(genrei)

  #if len(top_3_genre_names_per_user) == 0:
  #  return render_template("input_new.html", the_result = '')

  for i in range(3):
    if int(book_list[i]) == 0:
      continue

    user_input_list[int(book_list[i])] = 5

  print('Checking books; book list: ', book_list)
  print('..and genre list: ', top_3_genre_names_per_user)

  # Make title list for check later
  title_list = []

  for i in book_list:
      df_temp_in = df_ratings_cut.query('book_idx == %d' % i)
      book_id_in = df_temp_in.book_id.values[0]
      new_book_in = df_books.query('book_id == %d' % book_id_in)['title'].values
      title_list.append(new_book_in)

  new_ratings_mat = np.vstack([ratings_mat,user_input_list])

  # Calculate the user similarity and predictions
  user_similarity = fast_similarity(new_ratings_mat,kind='user')
  user_prediction = predict_fast_topk(new_ratings_mat, user_similarity, kind='user', k=30)
  user_prediction_0 = user_prediction[-1]

  pred_idxs_sorted = np.argsort(user_prediction_0)
  pred_idxs_sorted = pred_idxs_sorted[::-1]

  rec_v = []

  for i in range(20):

      ratings_book_idx = int(pred_idxs_sorted[i])
      df_temp = df_ratings_cut.query('book_idx == %d' % ratings_book_idx)
      book_id = df_temp.book_id.values[0]
      new_book = df_books.query('book_id == %d'%book_id)

      if new_book['title'].values[0] not in title_list: # and new_book['ratings_count'].values[0] < 2000000:

          url = new_book['image_url'].values[0] 
          book_url_id = "https://www.goodreads.com/book/show/"+str(new_book['goodreads_book_id'].values[0]) 
          title = new_book['title'].values[0] 
          genre_i = new_book['genre'].values[0] 
          author_i = new_book['authors'].values[0] 
          author = author_i.split(',')

          rec_v.append((url,book_url_id,title,author[0],genre_i))
 
  return render_template("output_new.html", recs = rec_v[:3], genres = top_3_genre_names_per_user, the_result = '')
