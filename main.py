import db_connections as db
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import json
import sample

def train_model():
    sample.model_elastic()
    print("the scoring model has trained successfully")

