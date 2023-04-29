#import 
import numpy as np
import pandas as pd
import re
import nltk
import base64
import tensorflow as tf
import tensorflow.compat.v1 as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from bing_image_downloader import downloader
import math, re, datetime as dt, glob
from urllib.parse import quote
from urllib.request import Request, urlopen
from google_images_download import google_images_download
from PIL import Image
from nltk.corpus import wordnet
import subprocess
import io
from base64 import encodebytes
from PIL import Image

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import ipywidgets as w
from IPython.display import display, IFrame
import pyspark
from pyspark.sql import SQLContext, functions, types
from pyspark.sql import Row
import matplotlib.pyplot as plt
# from hotel_recc import *
# %matplotlib inline

import pandas as pd
import ipywidgets as w
from IPython.display import display, IFrame
import math, re, numpy as np, pyspark, glob
from urllib.parse import quote
from urllib.request import Request, urlopen
from google_images_download import google_images_download
from pyspark.sql import SQLContext, functions, types
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Row
from geopy.geocoders import Nominatim

#initialize section
nltk.download('wordnet')
tf.disable_v2_behavior()
matplotlib.use('agg')


output = subprocess.check_output(['pwd'])
output_str = output.decode('utf-8').split('\n')
    


#code section
class Util(object):

    def read_data(self, folder):
        '''
        Function to read data required to
        build the recommender system
        '''
        print("Reading the data")
        ratings = pd.read_json(folder+"attraction_reviews.json",orient='records')
        attractions = pd.read_json(folder+"attractions.json",orient='records')
        return ratings, attractions

    def clean_subset(self, ratings, num_rows):
        '''
        Function to clean and subset the data according
        to individual machine power
        '''
        print("Extracting num_rows from ratings")
        temp = ratings.sort_values(by=['user_id'], ascending=True)
        ratings = temp.iloc[:num_rows, :]
        return ratings

    def preprocess(self, ratings):
        '''
        Preprocess data for feeding into the network
        '''
        print("Preprocessing the dataset")
        unique_att = ratings.attraction_id.unique()
        unique_att.sort()
        att_index = [i for i in range(len(unique_att))]
        rbm_att_df = pd.DataFrame(list(zip(att_index,unique_att)), columns =['rbm_att_id','attraction_id'])

        joined = ratings.merge(rbm_att_df, on='attraction_id')
        joined = joined[['user_id','attraction_id','rbm_att_id','rating']]
        readers_group = joined.groupby('user_id')

        total = []
        for readerID, curReader in readers_group:
            temp = np.zeros(len(ratings))
            for num, book in curReader.iterrows():
                temp[book['rbm_att_id']] = book['rating']/5.0
            total.append(temp)

        return joined, total

    def free_energy(self, v_sample, W, vb, hb):
        '''
        Function to compute the free energy
        '''
        wx_b = np.dot(v_sample, W) + hb
        vbias_term = np.dot(v_sample, vb)
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis = 1)
        return -hidden_term - vbias_term
    
class RBM(object):
    '''
    Class definition for a simple RBM
    '''
    def __init__(self, alpha, H, num_vis):

        self.alpha = alpha
        self.num_hid = H
        self.num_vis = num_vis # might face an error here, call preprocess if you do
        self.errors = []
        self.energy_train = []
        self.energy_valid = []

    def load_predict(self, filename, train, user):
        vb = tf.compat.v1.placeholder(tf.float32, [self.num_vis]) # Number of unique books
        hb = tf.compat.v1.placeholder(tf.float32, [self.num_hid]) # Number of features were going to learn
        W = tf.compat.v1.placeholder(tf.float32, [self.num_vis, self.num_hid])  # Weight Matrix
        v0 = tf.compat.v1.placeholder(tf.float32, [None, self.num_vis])
        
        prv_w =  np.load(output_str[0] + '/rbm_models/'+filename+'/w.npy')
        prv_vb = np.load(output_str[0] + '/rbm_models/'+filename+'/vb.npy')
        prv_hb = np.load(output_str[0] + '/rbm_models/'+filename+'/hb.npy')
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        print("Model restored")
        
        inputUser = [train[user]]
        
        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
        
        return rec, prv_w, prv_vb, prv_hb
        
    def calculate_scores(self, ratings, attractions, rec, user):
        '''
        Function to obtain recommendation scores for a user
        using the trained weights
        '''
        # Creating recommendation score for books in our data
        ratings["Recommendation Score"] = rec[0]

        """ Recommend User what books he has not read yet """
        # Find the mock user's user_id from the data
#         cur_user_id = ratings[ratings['user_id']

        # Find all books the mock user has read before
        visited_places = ratings[ratings['user_id'] == user]['attraction_id']
        visited_places

        # converting the pandas series object into a list
        places_id = visited_places.tolist()

        # getting the book names and authors for the books already read by the user
        places_names = []
        places_categories = []
        places_prices = []
        for place in places_id:
            places_names.append(
                attractions[attractions['attraction_id'] == place]['name'].tolist()[0])
            places_categories.append(
                attractions[attractions['attraction_id'] == place]['category'].tolist()[0])
            places_prices.append(
                attractions[attractions['attraction_id'] == place]['price'].tolist()[0])

        # Find all books the mock user has 'not' read before using the to_read data
        unvisited = attractions[~attractions['attraction_id'].isin(places_id)]['attraction_id']
        unvisited_id = unvisited.tolist()
        
        # extract the ratings of all the unread books from ratings dataframe
        unseen_with_score = ratings[ratings['attraction_id'].isin(unvisited_id)]

        # grouping the unread data on book id and taking the mean of the recommendation scores for each book_id
        grouped_unseen = unseen_with_score.groupby('attraction_id', as_index=False)['Recommendation Score'].max()
        
        # getting the names and authors of the unread books
        unseen_places_names = []
        unseen_places_categories = []
        unseen_places_prices = []
        unseen_places_scores = []
        for place in grouped_unseen['attraction_id'].tolist():
            unseen_places_names.append(
                attractions[attractions['attraction_id'] == place]['name'].tolist()[0])
            unseen_places_categories.append(
                attractions[attractions['attraction_id'] == place]['category'].tolist()[0])
            unseen_places_prices.append(
                attractions[attractions['attraction_id'] == place]['price'].tolist()[0])
            unseen_places_scores.append(
                grouped_unseen[grouped_unseen['attraction_id'] == place]['Recommendation Score'].tolist()[0])

        # creating a data frame for unread books with their names, authors and recommendation scores
        unseen_places = pd.DataFrame({
            'att_id' : grouped_unseen['attraction_id'].tolist(),
            'att_name': unseen_places_names,
            'att_cat': unseen_places_categories,
            'att_price': unseen_places_prices,
            'score': unseen_places_scores
        })

        # creating a data frame for read books with the names and authors
        seen_places = pd.DataFrame({
            'att_id' : places_id,
            'att_name': places_names,
            'att_cat': places_categories,
            'att_price': places_prices
        })

        return unseen_places, seen_places

    def export(self, unseen, seen, filename, user):
        '''
        Function to export the final result for a user into csv format
        '''
        # sort the result in descending order of the recommendation score
        sorted_result = unseen.sort_values(
            by='score', ascending=False)
        
        x = sorted_result[['score']].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler((0,5))
        x_scaled = min_max_scaler.fit_transform(x)
        
        sorted_result['score'] = x_scaled
        
        # exporting the read and unread books  with scores to csv files

        seen.to_csv(filename+'/user'+user+'_seen.csv')
        sorted_result.to_csv(filename+'/user'+user+'_unseen.csv')

def f(row):
    avg_cat_rat = dict()
    for i in range(len(row['category'])):
        if row['category'][i] not in avg_cat_rat:
            avg_cat_rat[row['category'][i]] = [row['rating'][i]]
        else:
            avg_cat_rat[row['category'][i]].append(row['rating'][i])
    for key,value in avg_cat_rat.items():
        avg_cat_rat[key] = sum(value)/len(value)
    return avg_cat_rat

def sim_score(row):
    score = 0.0
    match = 0
    col1 = row['cat_rat']
    col2 = row['user_data']
    for key, value in col2.items():
        if key in col1:
            match+=1
            score += (value-col1[key])**2
    if match != 0:
        return ((math.sqrt(score)/match) + (len(col2) - match))
    else:
        return 100

def get_recc(att_df, cat_rating):
    util = Util()
    epochs = 50
    rows = 40000
    alpha = 0.01
    H = 128
    batch_size = 16
    # print("hello1")
    dir= output_str[0] + '/etl/'
    # print('hello2')
    ratings, attractions = util.read_data(dir)
    # print(ratings)
    ratings = util.clean_subset(ratings, rows)
    rbm_att, train = util.preprocess(ratings)
    num_vis =  len(ratings)
    rbm = RBM(alpha, H, num_vis)
    
    joined = ratings.set_index('attraction_id').join(attractions[["attraction_id", "category"]].set_index("attraction_id")).reset_index('attraction_id')
    grouped = joined.groupby('user_id')
    category_df = grouped['category'].apply(list).reset_index()
    rating_df = grouped['rating'].apply(list).reset_index()
    cat_rat_df = category_df.set_index('user_id').join(rating_df.set_index('user_id'))
    cat_rat_df['cat_rat'] = cat_rat_df.apply(f,axis=1)
    cat_rat_df = cat_rat_df.reset_index()[['user_id','cat_rat']]
    
    cat_rat_df['user_data'] = [cat_rating for i in range(len(cat_rat_df))]
    cat_rat_df['sim_score'] = cat_rat_df.apply(sim_score, axis=1)
    user = cat_rat_df.sort_values(['sim_score']).values[0][0]
    
    print("Similar User: {u}".format(u=user))
    filename = "e"+str(epochs)+"_r"+str(rows)+"_lr"+str(alpha)+"_hu"+str(H)+"_bs"+str(batch_size)
    print(filename)
    reco, weights, vb, hb = rbm.load_predict(filename,train,user)
    unseen, seen = rbm.calculate_scores(ratings, attractions, reco, user)
    rbm.export(unseen, seen, output_str[0] + '/rbm_models/'+filename, str(user))
    return filename, user, rbm_att

def filter_df(filename, user, low, high, province, att_df):
    recc_df = pd.read_csv(output_str[0] + '/rbm_models/'+filename+'/user{u}_unseen.csv'.format(u=user), index_col=0)
    recc_df.columns = ['attraction_id', 'att_name', 'att_cat', 'att_price', 'score']
    recommendation = att_df[['attraction_id','name','category','city','latitude','longitude','price','province', 'rating']].set_index('attraction_id').join(recc_df[['attraction_id','score']].set_index('attraction_id'), how="inner").reset_index().sort_values("score",ascending=False)
    
    filtered = recommendation[(recommendation.province == province) & (recommendation.price >= low) & (recommendation.price >= low)]
    url = pd.read_json(output_str[0] + '/outputs/attractions_cat.json',orient='records')
    url['id'] = url.index
    with_url = filtered.set_index('attraction_id').join(url[['id','attraction']].set_index('id'), how="inner")
    return with_url

def get_image3(name):
    name = name.split(",")[0]
    try:
      downloader.download(name, limit=1,  output_dir=output_str[0] + '/images2/', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
      
      for filename in glob.glob(output_str[0] + "/images2/{name}/*jpg".format(name=name)) + glob.glob(output_str[0] + "/images2/{name}/*png".format(name=name)):
            return filename
    except:

      for filename in glob.glob(output_str[0] + "/images2/{name}/*jpg".format(name = name)):
            return filename


def top_recc(with_url, final):
    i=0
    while(1):
        first_recc = with_url.iloc[[i]]
        if(first_recc['name'].values.T[0] not in final['name']):
            final['name'].append(first_recc['name'].values.T[0])
            final['location'].append(first_recc[['latitude','longitude']].values.tolist()[0])
            final['price'].append(first_recc['price'].values.T[0])
            final['rating'].append(first_recc['rating'].values.T[0])

            image_location = get_image3(first_recc['name'].values.T[0]);
            if image_location is None:
                image_location = output_str[0] + '/Image_1.jpg';
            with open(image_location, 'rb') as f:
                img_data = f.read()
                encoded_img_data = base64.b64encode(img_data).decode('utf-8')
            final['image'].append(encoded_img_data)
            final['category'].append(first_recc['category'].values.T[0])
            return final
        else:
            i+=1

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

def find_closest(with_url, loc, tod, final):
    syns1 = wordnet.synsets("evening")
    syns2 = wordnet.synsets("night")
    evening = [word.lemmas()[0].name() for word in syns1] + [word.lemmas()[0].name() for word in syns2]
    distance = list()
    for i in with_url[['latitude','longitude']].values.tolist():
        distance.append(math.sqrt((loc[0]-i[0])**2 + (loc[1]-i[1])**2))
    with_dist = with_url
    with_dist["distance"] = distance
    sorted_d = with_dist.sort_values(['distance','price'], ascending=[True,False])
    if tod == "Evening":
        mask = sorted_d.name.apply(lambda x: any(j in x for j in evening))
    else:
        mask = sorted_d.name.apply(lambda x: all(j not in x for j in evening))
    final = top_recc(sorted_d[mask], final)
    return final


# calling the model
att_df = pd.read_json(output_str[0] + '/etl/attractions.json',orient='records')

import datetime

#main call function.
def predict_api_call(province, low, high, cat_rating, begin_date, end_date):
  filename, user, rbm_att = get_recc(att_df, cat_rating)
  with_url = filter_df(filename, user, low, high, province, att_df)
  final = dict()
  final['timeofday'] = []
  final['image'] = []
  final['name'] = []
  final['location'] = []
  final['price'] = []
  final['rating'] = []
  final['category'] = []

  for i in range(1, (end_date - begin_date).days + 2):
      for j in range(4):
          final['timeofday'].append('Morning')
      for j in range(4):
          final['timeofday'].append('Evening')

  for i in range(len(final['timeofday'])):
      if i % 8 == 0:
          final = top_recc(with_url, final)
      else:
          if len(final['location']) > i and len(final['timeofday']) > i:
              final = find_closest(with_url, final['location'][i], final['timeofday'][i], final)
          else:
              final = top_recc(with_url, final)
  return final

#----------------------------------------------------------------
#
#                   HOTEL RECOMMANDATION CODE
#
#----------------------------------------------------------------
def get_rating(x):
    val = x / 5
    if x >= 0 and x <= val:
        return 1
    elif x > val and x <= 2*val:
        return 2
    elif x > 2*val and x <= 3*val:
        return 3
    elif x > 3*val and x <= 4*val:
        return 4
    else:
        return 5

def amenities_rating(spark, amenities_pref, newh_df):
    pa_df = pd.DataFrame(amenities_pref,columns=["amenities_pref"])

    a_df = spark.createDataFrame(pa_df)
    a_df.createOrReplaceTempView('a_df')
    
    newh_df.createOrReplaceTempView('del_dup')
    newa_df  = spark.sql("SELECT * FROM newh_df INNER JOIN a_df WHERE newh_df.amenities=a_df.amenities_pref")

    ameni_comb = newa_df.groupBy(functions.col("id")).agg(functions.collect_list( functions.col("amenities")).alias("amenities"))
    
    amenities_len=ameni_comb.withColumn("ameni_len",functions.size(ameni_comb["amenities"])).orderBy(functions.col("ameni_len"), ascending=False)
    amenities_len.createOrReplaceTempView("amenities_len")

    ameni_df = spark.sql("SELECT a.id,h.amenities,a.ameni_len FROM del_dup h INNER JOIN amenities_len a WHERE h.id=a.id ORDER BY a.ameni_len DESC")
    
    find_rating = functions.udf(lambda a: get_rating(a), types.IntegerType())
    usr_rating = ameni_df.withColumn("rating",find_rating("ameni_len"))
    return usr_rating

def model_train(spark, usr_rating):
    ## Adding new user info to original dataset for training

    u_id_df = spark.read.json(output_str[0] + '/etl/u_id_df')
    u_id_df.createOrReplaceTempView('u_id_df')
    uid_count = u_id_df.select("user_id").distinct().count()

    usrid_df = usr_rating.withColumn("usr_id", functions.lit(uid_count)).join(u_id_df.select(["id","att_id"]), "id")
    usrid_final_df = usrid_df.select(usrid_df["usr_id"].alias("user_id"),usrid_df["att_id"].alias("att_id"),usrid_df["rating"].alias("user_rating"))
    return usrid_final_df


def get_hotel_recc(spark, usrid_s2):
    als_model = ALSModel.load(output_str[0] + "/mf_models/model_file")

    user = usrid_s2.select("user_id").distinct()
    recomm = als_model.recommendForUserSubset(user,50)
    recomm.createOrReplaceTempView('recomm')

    recomm_df  = spark.sql("SELECT user_id,explode(recommendations) as recommendations FROM recomm")

    get_attid = recomm_df.withColumn("att_id",functions.col("recommendations.att_id")).withColumn("rating",functions.col("recommendations.rating"))
    get_attid.createOrReplaceTempView("get_attid")
    
    u_id_df = spark.read.json(output_str[0] + '/etl/u_id_df')
    u_id_df.createOrReplaceTempView('u_id_df')
    u_tempdf = spark.sql("SELECT u_id_df.id FROM u_id_df INNER JOIN get_attid on u_id_df.att_id=get_attid.att_id")

    return u_tempdf

        
def get_top_amenities(spark):
    ## Reading file containing hotel details after removing duplicates
    del_dup = spark.read.json(output_str[0] + '/etl/del_dup')

    ## Reading file containing hotel details after removing duplicates and exploding amenities
    newh_df = spark.read.json(output_str[0] + '/etl/newh_df')

    del_dup.createOrReplaceTempView('del_dup')
    newh_df.createOrReplaceTempView('newh_df')

    ## Finding top 15 amentities to ask users to select inorder to provide hotel recommendations based on amenities chosen

    newh1_df  = spark.sql("SELECT amenities,COUNT(amenities) AS tot_count FROM newh_df GROUP BY amenities ORDER BY tot_count DESC")
    top_amenities = [x[0] for x in newh1_df.head(16) if x[0] != '']

    return top_amenities



def init_hotel_recc(place, amenities, spark):
#def init_hotel_recc(name, place, amenities_pref):
        ## Reading file containing hotel details after removing duplicates
    del_dup = spark.read.json(output_str[0] + '/etl/del_dup')
    place = place.lower()
    ## Reading file containing hotel details after removing duplicates and exploding amenities
    newh_df = spark.read.json(output_str[0] + '/etl/newh_df')

    del_dup.createOrReplaceTempView('del_dup')
    newh_df.createOrReplaceTempView('newh_df')

    usr_rating = amenities_rating(spark, amenities, newh_df)

    usrid_s2 = model_train(spark, usr_rating)
    u_tempdf = get_hotel_recc(spark, usrid_s2)


    hotel_df = del_dup.join(u_tempdf, "id").withColumn("address",functions.lower(functions.col("address")))
    user_location = place
    hotel_sugg = hotel_df.where(hotel_df.address.contains(user_location))

    print('user_location', user_location)
    print(hotel_sugg.count())
    recc = hotel_sugg.dropna().toPandas()
    get_top_amenities(spark)   
    # Commented out IPython magic to ensure Python compatibility.
    # %%capture
    final = dict()

    final['address'] = recc[:5]['address'].values.tolist()
    final['experience'] = recc[:5]['hotel_experience'].values.tolist()
    final['name'] = recc[:5]['hotel_name'].values.tolist()
    final['rating'] = recc[:5]['hotel_rating'].values.tolist()
    final['location'] = [i[1:-1] for i in recc[:5]['location'].values.tolist()]
    final['price'] = recc[:5]['price'].values.tolist()
    final['image'] = [getBase64Image(i) for i in recc[:5]['hotel_name'].values.tolist()]

    return final

def getBase64Image(name):
    image_location = get_image3(name);
    if image_location is None:
        image_location = output_str[0] + '/Image_1.jpg';
    with open(image_location, 'rb') as f:
        img_data = f.read()
        encoded_img_data = base64.b64encode(img_data).decode('utf-8')
        return encoded_img_data