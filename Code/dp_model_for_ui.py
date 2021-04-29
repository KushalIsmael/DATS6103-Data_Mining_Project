'''

#---------------------------------------------------------------------------------------------------------
FiveThirtyEight Non-Voters Dataset
#---------------------------------------------------------------------------------------------------------

'''

##### Import packages and data #####

import pandas as pd
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, plot_roc_curve, roc_auc_score

dir = str(Path(os.getcwd()).parents[0])
df = pd.read_csv(dir+'\\'+'nonvoters_data.csv', sep=',', header=0)


##### Exploratory data analysis ##### --------------------------------------------------------------------------

'''
print(df['Q1'].value_counts())
print(df['ppage'].value_counts())
print(df['educ'].value_counts())
print(df['race'].value_counts())
print(df['gender'].value_counts())
print(df['income_cat'].value_counts())
print(df['voter_category'].value_counts())
'''

#### Data Pre-Processing to prepare for modeling ##### --------------------------------------------------------------------------

# Rename columns to descriptive names
df.columns = ['RespId', 'weight',
              'q1_uscitizen',
              'q2_important_voting','q2_important_jury','q2_important_following','q2_important_displaying','q2_important_census',
              'q2_important_pledge','q2_important_military','q2_important_respect','q2_important_god','q2_important_protesting',
              'q3_statement_racism1','q3_statement_racism2','q3_statement_feminine',
              'q3_statement_msm','q3_statement_politiciansdontcare','q3_statement_besensitive',
              'q4_impact_officialsfed','q4_impact_officialsstate','q4_impact_officialslocal',
              'q4_impact_news','q4_impact_wallstreet','q4_impact_lawenforcement',
              'q5_electionmatters',
              'q6_officialsarelikeyou',
              'q7_governmentdesign',
              'q8_trust_presidency','q8_trust_congress','q8_trust_supremecourt','q8_trust_cdc','q8_trust_electedofficials',
              'q8_trust_fbicia','q8_trust_newsmedia','q8_trust_police','q8_trust_postalservice',
              'q9_politicalsystems_democracy','q9_politicalsystems_experts','q9_politicalsystems_strongleader','q9_politicalsystems_army',
              'q10_disability','q10_chronic_illness','q10_unemployed','q10_evicted',
              'q11_lostjob','q11_gotcovid','q11_familycovid',
              'q11_coviddeath','q11_worriedmoney','q11_quitjob',
              'q14_view_of_republicans',
              'q15_view_of_democrats',
              'q16_how_easy_vote',
              'q17_secure_votingmachines','q17_secure_paperballotsinperson','q17_secure_paperballotsmail','q17_secure_electronicvotesonline',
              'q18_votingsituations1','q18_votingsituations2','q18_votingsituations3','q18_votingsituations4','q18_votingsituations5',
              'q18_votingsituations6','q18_votingsituations7','q18_votingsituations8','q18_votingsituations9','q18_votingsituations10',
              'q19_get_more_voting1','q19_get_more_voting2','q19_get_more_voting3','q19_get_more_voting4','q19_get_more_voting5',
              'q19_get_more_voting6','q19_get_more_voting7','q19_get_more_voting8','q19_get_more_voting9','q19_get_more_voting10',
              'q20_currentlyregistered',
              'q21_plan_to_vote',
              'q22_whynotvoting_2020',
              'q23_which_candidate_supporting',
              'q24_preferred_voting_method',
              'q25_howcloselyfollowing_election',
              'q26_which_voting_category',
              'q27_didyouvotein18','q27_didyouvotein16','q27_didyouvotein14',
              'q27_didyouvotein12','q27_didyouvotein10','q27_didyouvotein08',
              'q28_whydidyouvote_past1','q28_whydidyouvote_past2','q28_whydidyouvote_past3','q28_whydidyouvote_past4',
              'q28_whydidyouvote_past5','q28_whydidyouvote_past6','q28_whydidyouvote_past7','q28_whydidyouvote_past8',
              'q29_whydidyounotvote_past1','q29_whydidyounotvote_past2','q29_whydidyounotvote_past3','q29_whydidyounotvote_past4','q29_whydidyounotvote_past5',
              'q29_whydidyounotvote_past6','q29_whydidyounotvote_past7','q29_whydidyounotvote_past8','q29_whydidyounotvote_past9','q29_whydidyounotvote_past10',
              'q30_partyidentification',
              'q31_republicantype',
              'q32_democratictype',
              'q33_closertowhichparty',
              'ppage', 'educ', 'race', 'gender', 'income_cat', 'voter_category'
              ]


# Drop irrelevant fields (US Citizen, responder ID, observation weight)
# Drop questions that were not asked to all participants (i.e. "why did you vote" to non-voters, "Republican type" for Democrats)
df.drop(['q1_uscitizen','q22_whynotvoting_2020',
              'q28_whydidyouvote_past1','q28_whydidyouvote_past2','q28_whydidyouvote_past3','q28_whydidyouvote_past4',
              'q28_whydidyouvote_past5','q28_whydidyouvote_past6','q28_whydidyouvote_past7','q28_whydidyouvote_past8',
              'q29_whydidyounotvote_past1','q29_whydidyounotvote_past2','q29_whydidyounotvote_past3','q29_whydidyounotvote_past4','q29_whydidyounotvote_past5',
              'q29_whydidyounotvote_past6','q29_whydidyounotvote_past7','q29_whydidyounotvote_past8','q29_whydidyounotvote_past9','q29_whydidyounotvote_past10',
              'q31_republicantype',
              'q32_democratictype',
              'q33_closertowhichparty',
         'q21_plan_to_vote',
         'q22_whynotvoting_2020',
         'RespId',
         'weight'
         ], axis=1, inplace=True)



# Replace "refused" answers (value of -1) with the demographic average for each group
# Step 1 - Replace -1 in certain columns with NaN
# Step 2 - Replace NaN with demographic average using groupby

# Create list of columns that need answer cleaning
# This isn't all the columns (some columns only had values of -1 and 1, which is fine)
replace_neg_one = [
              'q2_important_voting','q2_important_jury','q2_important_following','q2_important_displaying','q2_important_census',
              'q2_important_pledge','q2_important_military','q2_important_respect','q2_important_god','q2_important_protesting',
              'q3_statement_racism1','q3_statement_racism2','q3_statement_feminine',
              'q3_statement_msm','q3_statement_politiciansdontcare','q3_statement_besensitive',
              'q4_impact_officialsfed','q4_impact_officialsstate','q4_impact_officialslocal',
              'q4_impact_news','q4_impact_wallstreet','q4_impact_lawenforcement',
              'q5_electionmatters',
              'q6_officialsarelikeyou',
              'q7_governmentdesign',
              'q8_trust_presidency','q8_trust_congress','q8_trust_supremecourt','q8_trust_cdc','q8_trust_electedofficials',
              'q8_trust_fbicia','q8_trust_newsmedia','q8_trust_police','q8_trust_postalservice',
              'q9_politicalsystems_democracy','q9_politicalsystems_experts','q9_politicalsystems_strongleader','q9_politicalsystems_army',
              'q10_disability','q10_chronic_illness','q10_unemployed','q10_evicted',
              'q11_lostjob','q11_gotcovid','q11_familycovid',
              'q11_coviddeath','q11_worriedmoney','q11_quitjob',
              'q14_view_of_republicans',
              'q15_view_of_democrats',
              'q16_how_easy_vote',
              'q17_secure_votingmachines','q17_secure_paperballotsinperson','q17_secure_paperballotsmail','q17_secure_electronicvotesonline',
              'q18_votingsituations1','q18_votingsituations2','q18_votingsituations3','q18_votingsituations4','q18_votingsituations5',
              'q18_votingsituations6','q18_votingsituations7','q18_votingsituations8','q18_votingsituations9','q18_votingsituations10',
              'q20_currentlyregistered',
              'q24_preferred_voting_method',
              'q25_howcloselyfollowing_election',
              'q26_which_voting_category',
              'q27_didyouvotein18','q27_didyouvotein16','q27_didyouvotein14',
              'q27_didyouvotein12','q27_didyouvotein10','q27_didyouvotein08',
              'q30_partyidentification'
              ]

# Step 1 - Replace -1 or -1.0 values with NaN
# Values might be stored as int or float, so account for both
df[replace_neg_one] = df[replace_neg_one].replace(-1, np.nan)
df[replace_neg_one] = df[replace_neg_one].replace(-1.0, np.nan)

# Step 2 - Replace NaN with demographic mean
for x in replace_neg_one:
    df[x] = df[x].fillna(df.groupby(by=['educ', 'race', 'gender', 'income_cat'])[x].transform('mean'))

# Transform non-numeric categorical variables into numeric for model processing
le = LabelEncoder()
df['educ'] = le.fit_transform(df['educ'])
df['race'] = le.fit_transform(df['race'])
df['gender'] = le.fit_transform(df['gender'])
df['income_cat'] = le.fit_transform(df['income_cat'])
df['voter_category'] = le.fit_transform(df['voter_category'])

# Identify values of the target variable
#print(df['q23_which_candidate_supporting'].value_counts())

# For q23_which_candidate_supporting, value of 1 is Trump and value of 2 is Biden
# Drop unsure (value of 3) and refused to answer (value of -1) to set up our two-way classification
df = df[(df['q23_which_candidate_supporting'] == 1) | (df['q23_which_candidate_supporting'] == 2)]
# Label encode target variable
df['q23_which_candidate_supporting'] = le.fit_transform(df['q23_which_candidate_supporting'])

# Write a function that takes
# X, y, percent test in train-test, number of features in model
# Return model accuracy metrics, confusion matrix, feature importance, roc curve

def rf_model_visualize(df: pd.DataFrame, num_features: int, test_percent: float):
    # There are only 92 features available
    if (num_features < 1) or (num_features > 92):
        return ValueError # Need to change this later
    # We cannot test on 0 or 100 percent of our data
    if (test_percent < 0.01) or (test_percent > 0.99):
        return ValueError # Need to change this later

    # Go through modeling steps in this function
    # Start with getting X, y, and train-test split
    Xpre = df.drop(columns=['q23_which_candidate_supporting'], axis=1)
    ypre = df['q23_which_candidate_supporting']

    X_pre_train, X_pre_test, y_pre_train, y_pre_test = train_test_split(Xpre, ypre, test_size=test_percent, random_state=1918)

    # Fit the model
    rf_pre = RandomForestClassifier()
    rf_pre.fit(X_pre_train, y_pre_train)

    # Get the most important features
    importances = rf_pre.feature_importances_
    feat_imp = pd.Series(importances, X_pre_train.columns)
    feat_imp.sort_values(ascending=False, inplace=True)
    features_to_keep = feat_imp.index[0:num_features]

    # Re-fit with the slimmed down list
    X = Xpre[features_to_keep]
    y = ypre

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=1918)

    # Fit the model
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)

    # Output: accuracy metrics
    accuracy_score_value = accuracy_score(y_test, y_pred)

    # Output: confusion matrix
    conf = confusion_matrix(y_test, y_pred)

    # Output: ROC Curve
    roc_graph = plot_roc_curve(rf, X_test, y_test)

    # Output: ROC score
    roc_score_value = roc_auc_score(y_test, y_pred_proba[:, 1])

    # Output Feature importance
    imp_final = rf.feature_importances_
    feat_imp_final = pd.Series(imp_final, X_train.columns)
    feat_imp_final.sort_values(ascending=False, inplace=True)
    feature_importance_plot = plt.bar(x=feat_imp_final.index, height=feat_imp_final.values)

    return accuracy_score_value, conf, roc_graph, roc_score_value, feature_importance_plot


accuracy_score_value, conf, roc_graph, roc_score_value, feature_importance_plot = rf_model_visualize(df=df,
                                                                                                     num_features=25,
                                                                                                     test_percent=0.25)

print(accuracy_score_value)