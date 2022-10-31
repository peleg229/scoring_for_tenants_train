import pandas as pd
import json
import pickle


def score(x_f):
    feat_dict = json.loads(x_f)
    feat_dict["credit_score"] = feat_dict["credit_score"]/1000
    feat_dict["age"] = feat_dict["age"]/100
    mod = pd.DataFrame([feat_dict])
    model = pickle.load(open("tenant_score_ml_model", 'rb'))
    ans = model.predict(mod)
    if ans > 1:
        return 100
    elif 0.85 > ans:
        return ans*100+7
    else:
        return ans*100




x = """{"credit_score": 486,
        "eviction": 1,
        "criminal": 1,
        "age": 29}
    """
score = score(x)
