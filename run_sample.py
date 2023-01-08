import pandas as pd
import json
import pickle


def score(x_f):
    feat_dict = json.loads(x_f)
    feat_dict["credit_score"] = feat_dict["credit_score"]/1000
    feat_dict["age"] = feat_dict["age"]/85
    mod = pd.DataFrame([feat_dict])
    model = pickle.load(open("tenant_score_ml_model", 'rb'))
    ans = model.predict(mod)
    print("tenant scoring: ")
    if ans > 1:
        fans = 100
    elif ans < 0:
        fans = 20
    elif 0.85 > ans:
        fans = ans*100+7
    else:
        fans = ans*100
    if fans < 25:
        level = 1
    elif fans < 45:
        level = 2
    elif fans < 60:
        level = 3
    elif fans < 80:
        level = 4
    else:
        level = 5
    return {'level': level, 'score': fans}


x = """{applicant_id: 'BC5T2A09-ELE2-4HG0-AE74-4E2DF78A3E1D',
        "credit_score": 678,
        "eviction": 2,
        "criminal": 1,
        "age": 28}
    """
score = score(x)
