import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity

import joblib




def movies_data(Books, Users, Ratings):
    ## Merge rating column to the books table on ISBN number
    Ratings_with_name = Ratings.merge(Books, on='ISBN',how='left')
    Ratings_with_name.head()
    Book_Ratings_num = Ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    Book_Ratings_num.rename(columns={'Book-Rating':'Number_of_Ratings'}, inplace=True)
    Book_Ratings_Avg = Ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
    Book_Ratings_Avg.rename(columns={'Book-Rating':'Average_of_Ratings'}, inplace=True)

    ## The popularity data frame, which we will use to build Populatrity based recommendation system

    Pop_df = Book_Ratings_num.merge(Book_Ratings_Avg, on=['Book-Title'])
    ## more popularly recommenede books appear first.
    Pop_df = Pop_df[Pop_df['Number_of_Ratings']>250].sort_values('Average_of_Ratings',ascending=False)
    Pop_df = Pop_df.merge(Books, on=['Book-Title'],how='left').drop_duplicates('Book-Title')
    Pop_df = Pop_df[['Book-Title', 'Number_of_Ratings', 'Average_of_Ratings','ISBN', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
    # x contains User IDs who have given ratings more than 200 times:
    x = Ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    # Uder Ids of users who gave rating more than 200 times
    educated_users = x[x].index
    filtered_rating = Ratings_with_name[Ratings_with_name['User-ID'].isin(educated_users)]
    y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
    famous_books = y[y].index

    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0,inplace=True)

    similarity_scores = cosine_similarity(pt)
    Books.groupby('ISBN').count().reset_index()['Book-Title']

    return pt, similarity_scores


def recommendation_system(book_name,pt,similarity_scores ):
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key = lambda x : x[1], reverse=True)[1:6]
    Books = pd.read_csv('Books.csv')
    data = []
    for i in similar_items:
        item = []
        temp_df = Books[Books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-L'].values))
        
        data.append(item)
        
    return data



    

