from flask import Flask, request
from flask_cors import CORS
import constants
from model import predict_api_call, init_hotel_recc
import datetime
from pyspark.sql import SQLContext
import pyspark

app = Flask(__name__)
CORS(app)
sc=pyspark.SparkContext(appName="project")
spark = SQLContext(sc)

@app.route('/get_recommadations', methods=('GET', 'POST'))
def get_recommadations():
    if request.method == 'POST':
        province = request.get_json().get(constants.PROVINCE)
        low = request.get_json().get(constants.LOWEST_PRICE)
        high = request.get_json().get(constants.HIGHEST_PRICE)
        cat_rating = request.get_json().get(constants.CAT_RATING)
        begin_date = datetime.datetime.strptime(request.get_json().get(constants.TRIP_BEGIN_DATE), '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(request.get_json().get(constants.TRIP_END_DATE), '%Y-%m-%d').date()

    return predict_api_call(province, low, high, cat_rating, begin_date, end_date)

# @app.route('/get_ammenities')
# def get_ammenities():
#     return get_top_amenities()

@app.route('/get_hotel_recommandations', methods=('GET', 'POST'))
def get_hotel_recommandations():
    province = request.get_json().get('province').replace("_", " ")
    print("priasldfkjasl;dkf" ,  province)
    amenities = request.get_json().get('amenities')
    return init_hotel_recc(province, amenities, spark)
