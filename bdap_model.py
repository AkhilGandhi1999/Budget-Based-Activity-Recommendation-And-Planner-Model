import numpy as np
from IPython.display import display as d

# from attractions_recc import *
from IPython.display import display
import ipywidgets as w
import pandas as pd
import re

import nltk
nltk.download('wordnet')

import pandas as pd
import numpy as np
import random
import os

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

    def split_data(self, total_data):
        '''
        Function to split into training and validation sets
        '''
        print("Free energy required, dividing into train and validation sets")
        random.shuffle(total_data)
        n = len(total_data)
        print("Total size of the data is: {0}".format(n))
        size_train = int(n * 0.75)
        X_train = total_data[:size_train]
        X_valid = total_data[size_train:]
        print("Size of the training data is: {0}".format(len(X_train)))
        print("Size of the validation data is: {0}".format(len(X_valid)))
        return X_train, X_valid

    def free_energy(self, v_sample, W, vb, hb):
        '''
        Function to compute the free energy
        '''
        wx_b = np.dot(v_sample, W) + hb
        vbias_term = np.dot(v_sample, vb)
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis = 1)
        return -hidden_term - vbias_term


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


# from utils import Util
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from IPython.display import display
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

    def training(self, train, valid, user, epochs, batchsize, free_energy, verbose, filename):
        '''
        Function where RBM training takes place
        '''
        vb = tf.placeholder(tf.float32, [self.num_vis]) # Number of unique books
        hb = tf.placeholder(tf.float32, [self.num_hid]) # Number of features were going to learn
        W = tf.placeholder(tf.float32, [self.num_vis, self.num_hid])  # Weight Matrix
        v0 = tf.placeholder(tf.float32, [None, self.num_vis])

        print("Phase 1: Input Processing")
        _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
        # Gibb's Sampling
        h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
        print("Phase 2: Reconstruction")
        _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
        v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
        h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

        print("Creating the gradients")
        w_pos_grad = tf.matmul(tf.transpose(v0), h0)
        w_neg_grad = tf.matmul(tf.transpose(v1), h1)

        # Calculate the Contrastive Divergence to maximize
        CD = (w_pos_grad - w_neg_grad) / tf.cast(tf.shape(v0)[0], tf.float32)

        # Create methods to update the weights and biases
        update_w = W + self.alpha * CD
        update_vb = vb + self.alpha * tf.reduce_mean(v0 - v1, 0)
        update_hb = hb + self.alpha * tf.reduce_mean(h0 - h1, 0)

        # Set the error function, here we use Mean Absolute Error Function
        err = v0 - v1
        err_sum = tf.reduce_mean(err * err)

        # Initialize our Variables with Zeroes using Numpy Library
        # Current weight
        cur_w = np.zeros([self.num_vis, self.num_hid], np.float32)
        # Current visible unit biases
        cur_vb = np.zeros([self.num_vis], np.float32)

        # Current hidden unit biases
        cur_hb = np.zeros([self.num_hid], np.float32)

        # Previous weight
        prv_w = np.random.normal(loc=0, scale=0.01,
                                size=[self.num_vis, self.num_hid])
        # Previous visible unit biases
        prv_vb = np.zeros([self.num_vis], np.float32)

        # Previous hidden unit biases
        prv_hb = np.zeros([self.num_hid], np.float32)

        print("Running the session")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        print("Training RBM with {0} epochs and batch size: {1}".format(epochs, batchsize))
        print("Starting the training process")
        util = Util()
        for i in range(epochs):
            for start, end in zip(range(0, len(train), batchsize), range(batchsize, len(train), batchsize)):
                batch = train[start:end]
                cur_w = sess.run(update_w, feed_dict={
                                 v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_vb = sess.run(update_vb, feed_dict={
                                  v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_hb = sess.run(update_hb, feed_dict={
                                  v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                prv_w = cur_w
                prv_vb = cur_vb
                prv_hb = cur_hb

            if valid:
                etrain = np.mean(util.free_energy(train, cur_w, cur_vb, cur_hb))
                self.energy_train.append(etrain)
                evalid = np.mean(util.free_energy(valid, cur_w, cur_vb, cur_hb))
                self.energy_valid.append(evalid)
            self.errors.append(sess.run(err_sum, feed_dict={
                          v0: train, W: cur_w, vb: cur_vb, hb: cur_hb}))
            if verbose:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))
            elif i % 10 == 9:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))
        if not os.path.exists('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/rbm_models'):
            os.mkdir('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/rbm_models')
        filename = '/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/rbm_models/'+filename
        if not os.path.exists(filename):
            os.mkdir(filename)
        np.save(filename+'/w.npy', prv_w)
        np.save(filename+'/vb.npy', prv_vb)
        np.save(filename+'/hb.npy',prv_hb)
        
        if free_energy:
            print("Exporting free energy plot")
            self.export_free_energy_plot(filename)
        print("Exporting errors vs epochs plot")
        self.export_errors_plot(filename)
        inputUser = [train[user]]
        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
        return rec, prv_w, prv_vb, prv_hb

    def load_predict(self, filename, train, user):
        vb = tf.compat.v1.placeholder(tf.float32, [self.num_vis]) # Number of unique books
        hb = tf.compat.v1.placeholder(tf.float32, [self.num_hid]) # Number of features were going to learn
        W = tf.compat.v1.placeholder(tf.float32, [self.num_vis, self.num_hid])  # Weight Matrix
        v0 = tf.compat.v1.placeholder(tf.float32, [None, self.num_vis])
        
        prv_w = np.load('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/rbm_models/'+filename+'/w.npy')
        prv_vb = np.load('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/rbm_models/'+filename+'/vb.npy')
        prv_hb = np.load('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/rbm_models/'+filename+'/hb.npy')
        
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
        display(grouped_unseen.head())
        
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
#         print('The attractions visited by the user are:')
#         print(seen)
#         print('The attractions recommended to the user are:')
#         print(sorted_result)

    def export_errors_plot(self, filename):
        plt.plot(self.errors)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.savefig(filename+"/error.png")

    def export_free_energy_plot(self, filename):
        fig, ax = plt.subplots()
        ax.plot(self.energy_train, label='train')
        ax.plot(self.energy_valid, label='valid')
        leg = ax.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Free Energy")
        plt.savefig(filename+"/free_energy.png")

from bing_image_downloader import downloader

import pandas as pd
import numpy as np
import ipywidgets as w
from ipywidgets import HBox, VBox
from ipywidgets import Layout, widgets
from IPython.display import display, IFrame, HTML
# from utils import Util
# from rbm import RBM
import math, re, datetime as dt, glob
from urllib.parse import quote
from urllib.request import Request, urlopen
from google_images_download import google_images_download
from PIL import Image
from nltk.corpus import wordnet

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
    dir= '/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/etl/'
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
    rbm.export(unseen, seen, '/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/rbm_models/'+filename, str(user))
    return filename, user, rbm_att

def filter_df(filename, user, low, high, province, att_df):
    recc_df = pd.read_csv('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/rbm_models/'+filename+'/user{u}_unseen.csv'.format(u=user), index_col=0)
    recc_df.columns = ['attraction_id', 'att_name', 'att_cat', 'att_price', 'score']
    recommendation = att_df[['attraction_id','name','category','city','latitude','longitude','price','province', 'rating']].set_index('attraction_id').join(recc_df[['attraction_id','score']].set_index('attraction_id'), how="inner").reset_index().sort_values("score",ascending=False)
    
    filtered = recommendation[(recommendation.province == province) & (recommendation.price >= low) & (recommendation.price >= low)]
    url = pd.read_json('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/outputs/attractions_cat.json',orient='records')
    url['id'] = url.index
    with_url = filtered.set_index('attraction_id').join(url[['id','attraction']].set_index('id'), how="inner")
    return with_url

def get_image2(name):
    name = name.split(",")[0]
    try:
      downloader.download(name, limit=1,  output_dir='/content/images2/', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
        
      for filename in glob.glob("/content/images2/{name}/*jpg".format(name=name)) + glob.glob("/content/images2/{name}/*png".format(name=name)):
            return filename
    except:
      for filename in glob.glob("/content/images2/*jpg"):
            return filename

def get_image3(name):
    name = name.split(",")[0]
    try:
      downloader.download(name, limit=1,  output_dir='/content/images2/', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
        
      for filename in glob.glob("/content/images2/{name}/*jpg".format(name=name)) + glob.glob("/content/images2/{name}/*png".format(name=name)):
            return filename
    except:
      for filename in glob.glob("/content/images2/*jpg"):
            return filename

def top_recc(with_url, final):
    i=0
    while(1):
        print("this is i :", i)
        first_recc = with_url.iloc[[i]]
        # print('hello' + first_recc['name'].values.T[0])
        if(first_recc['name'].values.T[0] not in final['name']):
            final['name'].append(first_recc['name'].values.T[0])
            final['location'].append(first_recc[['latitude','longitude']].values.tolist()[0])
            final['price'].append(first_recc['price'].values.T[0])
            final['rating'].append(first_recc['rating'].values.T[0])
            final['image'].append(get_image3(first_recc['name'].values.T[0]))
            final['category'].append(first_recc['category'].values.T[0])
            return final
        else:
            i+=1

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

def final_output(days, final):
    time = ['MORNING', 'EVENING']
    fields = ['NAME', 'CATEGORY', 'LOCATION', 'PRICE', 'RATING']
    recommendations = ['Recommendation 1:', 'Recommendation 2:']

    box_layout = Layout(justify_content='space-between',
                        display='flex',
                        flex_flow='row', 
                        align_items='stretch',
                       )
    column_layout = Layout(justify_content='space-between',
                        width='75%',
                        display='flex',
                        flex_flow='column', 
                       )
    tab = []
    for i in range(days):
        print(final)
        print(type(final))
        images = final['image'][i*4:(i+1)*4]
        # print(images)
        # print("hello" + str(images))
        image = []
        # image = [open(str(i), "rb").read() for i in images]
        # image = [Image.open(i, mode='r') for i in images]
        for k in images :
          if k is None:
            image.append(open(str('/content/images2/best_of_niagara_falls_tour_from_niagara_falls/Image_1.jpg'), "rb").read())
          else :
            image.append(open(str(k), "rb").read())

        name = [re.sub('_',' ',i).capitalize() for i in final['name'][i*4:(i+1)*4]]
        category = [re.sub('_',' ',i).capitalize() for i in final['category'][i*4:(i+1)*4]]
        location = ["("+str(i[0])+","+str(i[1])+")" for i in final['location'][i*4:(i+1)*4]]
        price = [str(i) for i in final['price'][i*4:(i+1)*4]]
        rating = [str(i) for i in final['rating'][i*4:(i+1)*4]]
        tab.append(VBox(children=
                        [HBox(children=
                              [VBox(children=
                                    [widgets.HTML(value = f"<b><font color='orange'>{time[0]}</b>"),
                                     widgets.HTML(value = f"<b><font color='purple'>{recommendations[0]}</b>"),
                                     widgets.Image(value=image[0], format='jpg', width=300, height=400), 
                                     widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[0]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[0]}</b>", disabled=True),
                                     widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[0]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[0]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[0]}</b>", disabled=True)
                                    ], layout=column_layout), 
                                VBox(children=
                                    [widgets.HTML(value = f"<b><font color='orange'>{time[1]}</b>"), 
                                     widgets.HTML(value = f"<b><font color='purple'>{recommendations[0]}</b>"),
                                     widgets.Image(value=image[2], format='jpg', width=300, height=400), 
                                     widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[2]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[2]}</b>", disabled=True),
                                     widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[2]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[2]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[2]}</b>", disabled=True)
                                    ], layout=column_layout)
                              ], layout=box_layout),

                         HBox(children=
                              [VBox(children=
                                    [widgets.HTML(value = f"<b><font color='purple'>{recommendations[1]}</b>"),
                                     widgets.Image(value=image[1], format='jpg', width=300, height=400), 
                                     widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[1]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[1]}</b>", disabled=True),
                                     widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[1]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[1]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[1]}</b>", disabled=True)
                                    ], layout=column_layout), 
                                VBox(children=
                                    [widgets.HTML(value = f"<b><font color='purple'>{recommendations[1]}</b>"),
                                     widgets.Image(value=image[3], format='jpg', width=300, height=400), 
                                     widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[3]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[3]}</b>", disabled=True),
                                     widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[3]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[3]}</b>", disabled=True), 
                                     widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[3]}</b>", disabled=True)
                                    ], layout=column_layout),
                              ], layout=box_layout)

                        ]))

    tab_recc = widgets.Tab(children=tab)
    for i in range(len(tab_recc.children)):
        tab_recc.set_title(i, str('Day '+ str(i+1)))
    print(type(tab_recc))
    return tab_recc

att_df = pd.read_json('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/etl/attractions.json',orient='records')

import datetime

user_name = "akhil"
province = "british_columbia"
low = 115.0
high = 205.0
cat_rating = {'private_&_custom_tours': 4.0, 'luxury_&_special_occasions': 3.0, 'sightseeing_tickets_&_passes': 1.0, 'multi-day_&_extended_tours': 3.0, 'walking_&_biking_tours': 1.0}
begin_date = datetime.date(2023,4,4)
end_date = datetime.date(2023,4,5)

# uname = w.Text(description="User Name")
# print(type(uname))
# print(uname)
# place = w.Text(value = 'Province',  description="Destination")
# print(place)
# print(type(place))
# budget = w.IntRangeSlider(min=att_df.price.min(), max=att_df.price.max(),step=10,value=[att_df.price.min(),att_df.price.max()],description="Budget")
# v1 = w.VBox([uname,place,budget])
# start = w.DatePicker(description='Start Date',disabled=False)
# end = w.DatePicker(description='End Date',disabled=False)
# v2 = w.VBox([start,end])
# out = w.HBox([v1,v2])
# display(out)

# category_df = att_df.groupby('category').size().reset_index().sort_values([0],ascending=False)[:20]
# categories = list(category_df.category.values)
# cat_rat = dict()
# def on_button_clicked(b):
#     if b.description in cat_rat:
#         return
#     print(b.description)
#     slider = w.IntSlider(min=0,max=5,step=1,description='Rate')
#     display(slider)
#     cat_rat[b.description] = slider
#     if(len(cat_rat) < 5):
#         print("Rate {x} more!\n".format(x=5-len(cat_rat)))
    
# but_layout = w.Layout(width='100%', height='100px')
# but_items = [w.Button(description=c, layout=but_layout) for c in categories]
# on_clk = [item.on_click(on_button_clicked) for item in but_items]
# c1 = w.VBox([i for i in but_items[:4]])
# print(c1)
# print(type(c1))
# c2 = w.VBox([i for i in but_items[4:8]])
# c3 = w.VBox([i for i in but_items[8:12]])
# c4 = w.VBox([i for i in but_items[12:16]])
# c5 = w.VBox([i for i in but_items[16:]])
# buttons = w.HBox([c1,c2,c3,c4,c5])
# print("Select and rate atleast 5 categories and rate them:")
# display(buttons)

# user_name = re.sub(' ','_',uname.value.lower())
# print(user_name)
# print(type(user_name))
# province = re.sub(' ','_',place.value.lower())
# print(province)
# print(type(province))
# (low,high) = tuple([float(i) for i in budget.value])
# print(low)
# print(type(low))
# print(high)
# print(type(high))
# begin_date = start.value
# print(begin_date)
# print(type(begin_date))
# end_date = end.value
# print(end_date)
# print(type(end_date))
# cat_rating = dict()
# for key, value in cat_rat.items():
#     cat_rating[key] = float(value.value)
#     print(float(value.value))

# print(cat_rating)
# print(type(cat_rating))

def predict_api_call():
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
      for j in range(2):
          final['timeofday'].append('Morning')
      for j in range(2):
          final['timeofday'].append('Evening')

  for i in range(len(final['timeofday'])):
      if i % 4 == 0:
          final = top_recc(with_url, final)
          print('i is ', i)
      else:
          if len(final['location']) > i and len(final['timeofday']) > i:
              final = find_closest(with_url, final['location'][i], final['timeofday'][i], final)
              print('i is ', i)
          else:
              final = top_recc(with_url, final)
              print('i is ', i)
  days = (end_date - begin_date).days + 1
  display(final_output(days,final))

!pip install pyngrok

# Commented out IPython magic to ensure Python compatibility.
!mkdir -p /drive/MyDrive/ngrok-ssh
# %cd /drive/MyDrive/ngrok-ssh
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip -O ngrok-stable-linux-amd64.zip
!unzip -u ngrok-stable-linux-amd64.zip
!cp /drive/MyDrive/ngrok-ssh/ngrok /ngrok
!chmod +x /ngrok

!/ngrok authtoken 2Nz9vW47rEVWbsNjCMVw0XbHKC2_5JsERriJQpXcQ4pV1KAmw

!pkill ngrok

import threading

from flask import Flask
from pyngrok import ngrok

app = Flask(__name__)
port = 5000

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

# ... Update inbound traffic via APIs to use the public-facing ngrok URL


# Define Flask routes
@app.route("/track")
def index():
    predict_api_call()
    return "Hello from Colab!"

# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()

"""#Hotel"""

!pip install pyspark

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
    
## Finding number of amenities present in hotels that user likes
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
    u_id_df = spark.read.json('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/etl/u_id_df')
    u_id_df.createOrReplaceTempView('u_id_df')
    
    uid_count = u_id_df.select("user_id").distinct().count()

    usrid_df = usr_rating.withColumn("usr_id", functions.lit(uid_count)).join(u_id_df.select(["id","att_id"]), "id")

    usrid_final_df = usrid_df.select(usrid_df["usr_id"].alias("user_id"),usrid_df["att_id"].alias("att_id"),usrid_df["rating"].alias("user_rating"))

    org_df = u_id_df.select("user_id","att_id","user_rating")

    (usrid_s1, usrid_s2) = usrid_final_df.randomSplit([0.1,0.9])

    comb_df = org_df.union(usrid_s1)
    
    ## Model training and evaluation
    (training,validation) = comb_df.randomSplit([0.8,0.2])

    ranks=[4,8,12]
    error = 20000
    errors = []
    for i in ranks:
        als = ALS(maxIter= 5,regParam= 0.01,rank=i,userCol="user_id",itemCol="att_id", ratingCol="user_rating", coldStartStrategy="drop")
        model = als.fit(training)
        predictions = model.transform(validation)
        evaluator = RegressionEvaluator(metricName="rmse",labelCol="user_rating",predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    errors.append(rmse)
    if rmse < error:
        model.write().overwrite().save("mf_models/model_file")
        rank = i
        error = rmse
    
    return rank, error, errors, usrid_s2

def get_hotel_recc(spark, usrid_s2):
    als_model = ALSModel.load("mf_models/model_file")

    user = usrid_s2.select("user_id").distinct()
    recomm = als_model.recommendForUserSubset(user,50)
    recomm.createOrReplaceTempView('recomm')

    recomm_df  = spark.sql("SELECT user_id,explode(recommendations) as recommendations FROM recomm")

    get_attid = recomm_df.withColumn("att_id",functions.col("recommendations.att_id")).withColumn("rating",functions.col("recommendations.rating"))
    get_attid.createOrReplaceTempView("get_attid")
    
    u_id_df = spark.read.json('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/etl/u_id_df')
    u_id_df.createOrReplaceTempView('u_id_df')
    u_tempdf = spark.sql("SELECT u_id_df.id FROM u_id_df INNER JOIN get_attid on u_id_df.att_id=get_attid.att_id")
    
    return u_tempdf

def get_image(name):
    name = re.sub(' ','_',name)
    response = google_images_download.googleimagesdownload()
    args_list = ["keywords", "keywords_from_file", "prefix_keywords", "suffix_keywords",
             "limit", "format", "color", "color_type", "usage_rights", "size",
             "exact_size", "aspect_ratio", "type", "time", "time_range", "delay", "url", "single_image",
             "output_directory", "image_directory", "no_directory", "proxy", "similar_images", "specific_site",
             "print_urls", "print_size", "print_paths", "metadata", "extract_metadata", "socket_timeout",
             "thumbnail", "language", "prefix", "chromedriver", "related_images", "safe_search", "no_numbering",
             "offset", "no_download"]
    args = {}
    for i in args_list:
        args[i]= None
    args["keywords"] = name
    args['limit'] = 1
    params = response.build_url_parameters(args)
    url = 'https://www.google.com/search?q=' + quote(name) + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch' + params + '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
    try:
        response.download(args)
        for filename in glob.glob("downloads/{name}/*jpg".format(name=name))+glob.glob("downloads/{name}/*png".format(name=name)):
            return filename
    except:
        for filename in glob.glob("downloads/*jpg"):
            return filename

def hotel_image(name):
    name = name.split(",")[0]
    try:
      downloader.download(name, limit=1,  output_dir='/content/images2/', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
        
      for filename in glob.glob("/content/images2/{name}/*jpg".format(name=name)) + glob.glob("/content/images2/{name}/*png".format(name=name)):
            return filename
    except:
      for filename in glob.glob("/content/images2/*jpg"):
            return filename
        
def get_hotel_output(days, final):
    fields = ['NAME', 'PRICE', 'RATING', 'EXPERIENCE','LOCATION', 'ADDRESS', "AMENITIES"]
    recommendations = ['Recommendation']

    box_layout = w.Layout(justify_content='space-between',
                        display='flex',
                        flex_flow='row', 
                        align_items='stretch',
                       )
    column_layout = w.Layout(justify_content='space-between',
                        width='75%',
                        display='flex',
                        flex_flow='column', 
                       )
    tab = []
    for i in range(len(final['name'])):
        image = open(final['image'][i], "rb").read()
        name = final['name'][i]
        price= final['price'][i]
        rating= final['rating'][i]
        experience= final['experience'][i]
        loc=final['location'][i]
        address=final['address'][i]
        amenities=final['amenities'][i]
        tab.append(w.VBox(children=
                        [
                         w.Image(value=image, format='jpg', width=300, height=400),
                         w.HTML(description=fields[0], value=f"<b><font color='black'>{name}</b>", disabled=True),
                         w.HTML(description=fields[1], value=f"<b><font color='black'>{price}</b>", disabled=True),
                         w.HTML(description=fields[2], value=f"<b><font color='black'>{rating}</b>", disabled=True), 
                         w.HTML(description=fields[3], value=f"<b><font color='black'>{experience}</b>", disabled=True), 
                         w.HTML(description=fields[4], value=f"<b><font color='black'>{loc}</b>", disabled=True),
                         w.HTML(description=fields[5], value=f"<b><font color='black'>{address}</b>", disabled=True)
                        ], layout=column_layout))

    tab_recc = w.Tab(children=tab)
    for i in range(len(tab_recc.children)):
        tab_recc.set_title(i, str('Hotel '+ str(i+1)))
    return tab_recc

## Ipywidgets to get user name and destination
name = w.Text(description="User Name")
place = w.Text(description="Destination")
v1 = w.VBox([name,place])

## Ipywidgets to get start and end date of the trip

start = w.DatePicker(description='Start Date',disabled=False)
end = w.DatePicker(description='End Date',disabled=False)
v2 = w.VBox([start,end])

out = w.HBox([v1,v2])
display(out)

sc=pyspark.SparkContext(appName="project")
spark = SQLContext(sc)

## Reading file containing hotel details after removing duplicates
del_dup = spark.read.json('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/etl/del_dup')

## Reading file containing hotel details after removing duplicates and exploding amenities
newh_df = spark.read.json('/content/drive/MyDrive/big_data/Intelligent-Travel-Recommendation-System/etl/newh_df')

del_dup.createOrReplaceTempView('del_dup')
newh_df.createOrReplaceTempView('newh_df')

## Finding top 15 amentities to ask users to select inorder to provide hotel recommendations based on amenities chosen
newh1_df  = spark.sql("SELECT amenities,COUNT(amenities) AS tot_count FROM newh_df GROUP BY amenities ORDER BY tot_count DESC")
top_amenities = [x[0] for x in newh1_df.head(16) if x[0] != '']

## Getting user's amenity preferences
amenities_pref = []
def on_button_clicked(b):
    if b.description in amenities_pref:
        return
    print(b.description)
    amenities_pref.append(b.description)
    if(len(amenities_pref) < 5):
        print("Select {x} more!\n".format(x=5-len(amenities_pref)))
        
but_layout = w.Layout(width='100%', height='100px')
but_items = [w.Button(description=c, layout=but_layout) for c in top_amenities]
on_clk = [item.on_click(on_button_clicked) for item in but_items]
r1 = w.VBox([i for i in but_items[:5]])
r2 = w.VBox([i for i in but_items[5:10]])
r3 = w.VBox([i for i in but_items[10:]])
buttons = w.HBox([r1,r2,r3])
print("Select atleast 5 amenities:")
display(buttons)

usr_rating = amenities_rating(spark, amenities_pref, newh_df)
rank, error, errors, usrid_s2 = model_train(spark, usr_rating)
print("best rank : ",rank)
print("best RMSE:" +str(error))
plt.bar([4,8,12], height=errors)

u_tempdf = get_hotel_recc(spark, usrid_s2)
hotel_df = del_dup.join(u_tempdf, "id").withColumn("address",functions.lower(functions.col("address")))
user_location = place.value.lower()
hotel_sugg = hotel_df.where(hotel_df.address.contains(user_location))
recc = hotel_sugg.dropna().toPandas()

# Commented out IPython magic to ensure Python compatibility.
# %%capture
final = dict()
final['address'] = recc[:5]['address'].values.tolist()
final['amenities'] = recc[:5]['amenities'].values.T.tolist()
final['experience'] = recc[:5]['hotel_experience'].values.tolist()
final['name'] = recc[:5]['hotel_name'].values.tolist()
final['rating'] = recc[:5]['hotel_rating'].values.tolist()
final['location'] = [i[1:-1] for i in recc[:5]['location'].values.tolist()]
final['price'] = recc[:5]['price'].values.tolist()
final['image'] = [hotel_image(i) for i in recc[:5]['hotel_name'].values.tolist()]


days = (end.value - start.value).days
tab_recc = get_hotel_output(days, final)
display(tab_recc)

