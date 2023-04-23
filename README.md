Model File


input format for POST request:
{
    "province": "british_columbia",
    "low": 115.0,
    "high": 205.0,
    "cat_rating": {"private_&_custom_tours": 4.0, "luxury_&_special_occasions": 3.0, "sightseeing_tickets_&_passes": 1.0, "multi-day_&_extended_tours": 3.0, "walking_&_biking_tours": 1.0},
    "begin_date": "2023-04-05",
    "end_date": "2023-04-06"   
}
For hotel recommandations:
http://127.0.0.1:5001/hotels/get_hotel_recommandations
{
    "province": "british columbia",
    "amenities": [" Suites", " Wheelchair Access", " Microwave", " Breakfast included", "Laundry Service"]
}

running the flask app:
    flask --app app --debug -p 5000
