
import db_connections as db
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
import json


def cal_tenant(df):
    if len(df) < 2:
        df_date = df.loc[df['tenant_status'] == 'Inactive']
        if len(df_date) > 0:
            inactive_date = df_date['modified_date'].min()
        else:
            inactive_date = datetime.date.today()
        return 0, inactive_date
    df['dif'] = df['modified_date'].diff().array.days*-1
    new_df = df.loc[df["is_delinquent"] == True]
    df_date = df.loc[df['tenant_status'] == 'Inactive']
    ans = new_df['dif'].sum()
    if len(df_date) > 0:
        inactive_date = df_date['modified_date'].min()
    else:
        inactive_date = datetime.date.today()
    return ans, inactive_date


def score():
    query = """
    select tenant_id, modified_date::date, is_delinquent, date_added::date, tenant_status
    from "Tenants_History"
    union
    (select tenant_id, now()::date as modified_date, is_delinquent,date_added::date, tenant_status
    from "Tenants")
    order by tenant_id, modified_date desc;
    """
    df = db.importDataFromPG(query)
    ans_df = pd.DataFrame()
    ans_df['tenant_id'] = df['tenant_id'].drop_duplicates()
    ans_df['score'] = 0
    for index, row in df.drop_duplicates(subset=['tenant_id']).iterrows():
        if row['date_added'] is None:
            continue
        time_delinquency, inactive_date = cal_tenant(df.loc[df['tenant_id'] == row['tenant_id']])
        total_time = (inactive_date - (row['date_added'])).days
        score_delinquency = 100 - ((time_delinquency/total_time)*100)
        score_time = (total_time/(365*3))*100
        score_ans = score_time*0.23 + score_delinquency*0.77
        ans_df.loc[ans_df['tenant_id'] == row['tenant_id'], 'score'] = score_ans
    return ans_df


def logistic_sort():
    scores = score()
    scores['is_good'] = 0
    scores.loc[scores['score'] >= 74, 'is_good'] = 1
    return scores


def features():
    df_score = logistic_sort()
    query = """
    select tenant_id, "Applicants".additional_info additional_info,"Applications".additional_info additional_info_applictions, transunion_response, birthday::date birthday,status,reason
    from "Applicants" join "Applications" on "Applicants".application_id = "Applications".application_id
                      join "Tenants" on "Tenants".application_id = "Applicants".application_id
    where main_applicant ='true'
    union
    (select '000000-0000-0000-9876-89678765' as tenant_id, "Applicants".additional_info additional_info, "Applications".additional_info additional_info_applictions, transunion_response, birthday::date birthday, status, reason
    from "Applicants" join "Applications" on "Applicants".application_id = "Applications".application_id
    where main_applicant = 'true'
    and ("Applications".status in ('rejected', 'savedAsCandidate'))
    and ("Applications".reason not in ('none', 'other')))
    """
    df = db.importDataFromPG(query)
    df_final = pd.merge(df_score, df, how="right", on='tenant_id')
    df_final['is_employed'] = True
    df_final['self_employed'] = False
    df_final['pets'] = False
    df_final['salary'] = 2000
    df_final['additional_income'] = 0
    df_final['credit_score'] = 400
    df_final['eviction'] = 0
    df_final['criminal'] = 0
    df_final['age'] = 30
    df_final.loc[df['status'] == 'rejected', 'score'] = 20
    df_final.loc[df['status'] == 'rejected', 'is_good'] = 0
    df_final.loc[df['status'] == 'savedAsCandidate', 'score'] = 70
    df_final.loc[df['status'] == 'savedAsCandidate', 'is_good'] = 1

    # data preprocess - preparing the data
    for index, row in df_final.iterrows():
        if row['additional_info'] is not None and row['additional_info'] != 'null' and json.loads(row['additional_info'])['employmentAndIncome'] is not None:
            df_final['is_employed'][index] = json.loads(row['additional_info'])['employmentAndIncome']['isEmployed']
            df_final['self_employed'][index] = json.loads(row['additional_info'])['employmentAndIncome']['isSelfEmployed']
            if json.loads(row['additional_info'])['employmentAndIncome']['isEmployed'] is True:
                df_final['salary'][index] = json.loads(row['additional_info'])['employmentAndIncome']['employee'][0]['salary']
            elif json.loads(row['additional_info'])['employmentAndIncome']['isSelfEmployed'] is True:
                try:
                    df_final['salary'][index] = json.loads(row['additional_info'])['employmentAndIncome']['selfEmployed'][0]['salary']
                except:
                    df_final['salary'][index] = 3000
            if json.loads(row['additional_info'])['employmentAndIncome']['hasAdditionalIncome'] is True:
                df_final['additional_income'][index] = json.loads(row['additional_info'])['employmentAndIncome']['additionalIncome'][0]['income']
        if row['transunion_response'] is not None and row['transunion_response'] != 'null' and json.loads(row['transunion_response'])['application'] != None:
            df_final['credit_score'][index] = json.loads(row['transunion_response'])['application']['scoreResult']['applicationScore']
            if json.loads(row['transunion_response'])['application']['scoreResult']['applicationScore'] == '' or json.loads(row['transunion_response'])['application']['scoreResult']['applicationScore'] == None:
                df_final['credit_score'][index] = 543
            df_final['eviction'][index] = json.loads(row['transunion_response'])['application']['scoreResult']['evictionRecommendation']
            df_final['criminal'][index] = json.loads(row['transunion_response'])['application']['scoreResult']['criminalRecommendation']
        df_final['age'][index] = (datetime.datetime.today().year - row['birthday'].year)
        df_final['pets'] = json.loads(row['additional_info_applictions'])['hasPets']
    df_final.loc[df_final['self_employed'] == True, 'self_employed'] = 1
    df_final.loc[df_final['self_employed'] == False, 'self_employed'] = 0
    df_final.loc[df_final['is_employed'] == True, 'is_employed'] = 1
    df_final.loc[df_final['is_employed'] == False, 'is_employed'] = 0
    df_final.loc[df_final['pets'] == True, 'pets'] = 1
    df_final.loc[df_final['pets'] == False, 'pets'] = 0
    df_final = df_final.astype({'criminal': 'int'})
    df_final.loc[df_final['criminal'] > 1, 'criminal'] = 0
    df_final = df_final.astype({'credit_score': 'int'})
    df_final['credit_score'] = (df_final['credit_score'] - df_final['credit_score'].min())/500
    df_final['age'] = df_final['age']/85
    df_final = df_final.astype({'eviction': 'int'})
    return df_final[['credit_score', 'eviction', 'criminal', 'age', 'is_good']]


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def calc_clustering_measures(total_predictions, total_labels):
    if len(total_labels) > 0:
        results = confusion_matrix(total_labels, total_predictions)
        print(results)

        print("Detailed classification report:")
        print()
        print(classification_report(total_labels, total_predictions, digits=4))
        print("Accuracy:", accuracy_score(total_labels, total_predictions))
        print()
    else:
        print("Successfully ended running over the data")



def model_elastic():
    # preparing the data
    df = features()
    X = df.loc[:, ~df.columns.isin(["is_good"])]
    y = df["is_good"]
    # build the ml model
    elastic = ElasticNet(normalize=True)
    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    model = GridSearchCV(estimator=elastic,
                          param_grid={"alpha":np.logspace(-5, 2, 8),
                                      "l1_ratio":[0, .2, .4, .6, .8, 1]},
                          scoring =ftwo_scorer, n_jobs = 1, refit = True, cv = 10)
    model.fit(X, y)

    # data = df.values
    # # X, y = data[:, :-1], data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # testing
    y_pred = model.predict(x_test)
    y_test = y_test.tolist()
    precision = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    # turn the results to binary
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    # calculate precision and recall
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            precision = precision + 1
            if y_pred[i] == 1:
                tp = tp+1
            if y_pred[i] == 0:
                fp = fp+1
        if y_pred[i] == 0 and y_test[i] == 1:
            fn = fn + 1
        if y_test[i] == 0 and y_pred[i] == 1:
            tn = tn+1
    p = precision/len(y_pred)
    print('precision: ' + str(p))
    p = tp/(tp+tn)
    r = tp/(tp+fn)
    print('precision_good_tenants: ' + str(p))
    print('recall_good_tenants: : ' + str(r))
    calc_clustering_measures(y_pred, y_test)
    #pickle.dump(model, open(os.path.join("../pythonProject10/tenant_score_ml_model"), 'wb'))
    return score, y_pred



model_elastic()




def scoring(y):
    if y > 0.5:
        return 1
    else:
        return 0