'''

Load in the dataset.
Clean up variables as needed.
Create correlation matrix.
Do stepwise selection of variables.

'''

import pandas as pd
import os
from pathlib import Path
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

dir = str(Path(os.getcwd()).parents[0])

df = pd.read_csv(dir+'\\'+'nonvoters_data.csv', sep=',', header=0)
print(df.head)
initial_cols = df.columns
print([x for x in df.columns])

print(df['Q1'].value_counts())
print(df['ppage'].value_counts())
print(df['educ'].value_counts())
print(df['race'].value_counts())
print(df['gender'].value_counts())
print(df['income_cat'].value_counts())
print(df['voter_category'].value_counts())


df.columns = ['RespId', 'weight', 
              'q1_uscitizen',
              'q2_important_voting','q2_important_jury','q2_important_following','q2_important_displaying','q2_important_census',
              'q2_important_pledge','q2_important_military','q2_important_respect','q2_important_god','q2_important_protesting',
              'q3_statement_racism1`','q3_statement_racism2','q3_statement_feminine',
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

# TO DO
# Transform plan to vote variable into yes or no/notsure
print(df['q21_plan_to_vote'].value_counts())
print(df['q23_which_candidate_supporting'].value_counts())
print(df['q30_partyidentification'].value_counts())


# Drop US citizen question since they are all US Citizens
# Drop questions about past voting behavior since that is very predictive (or try with and without)
'''
df.drop(['q1_uscitizen','q20_currentlyregistered','q22_whynotvoting_2020',
              'q23_which_candidate_supporting','q26_which_voting_category',
              'q27_didyouvotein18','q27_didyouvotein16','q27_didyouvotein14',
              'q27_didyouvotein12','q27_didyouvotein10','q27_didyouvotein08',
              'q28_whydidyouvote_past1','q28_whydidyouvote_past2','q28_whydidyouvote_past3','q28_whydidyouvote_past4',
              'q28_whydidyouvote_past5','q28_whydidyouvote_past6','q28_whydidyouvote_past7','q28_whydidyouvote_past8',
              'q29_whydidyounotvote_past1','q29_whydidyounotvote_past2','q29_whydidyounotvote_past3','q29_whydidyounotvote_past4','q29_whydidyounotvote_past5',
              'q29_whydidyounotvote_past6','q29_whydidyounotvote_past7','q29_whydidyounotvote_past8','q29_whydidyounotvote_past9','q29_whydidyounotvote_past10',
              'q31_republicantype',
              'q32_democratictype',
              'q33_closertowhichparty',
         'voter_category',
         'RespId',
         'weight'
         ], axis=1, inplace=True)
'''


df.drop(['q1_uscitizen','q22_whynotvoting_2020',
              'q28_whydidyouvote_past1','q28_whydidyouvote_past2','q28_whydidyouvote_past3','q28_whydidyouvote_past4',
              'q28_whydidyouvote_past5','q28_whydidyouvote_past6','q28_whydidyouvote_past7','q28_whydidyouvote_past8',
              'q29_whydidyounotvote_past1','q29_whydidyounotvote_past2','q29_whydidyounotvote_past3','q29_whydidyounotvote_past4','q29_whydidyounotvote_past5',
              'q29_whydidyounotvote_past6','q29_whydidyounotvote_past7','q29_whydidyounotvote_past8','q29_whydidyounotvote_past9','q29_whydidyounotvote_past10',
              'q31_republicantype',
              'q32_democratictype',
              'q33_closertowhichparty',
         'RespId',
         'weight'
         ], axis=1, inplace=True)

le = LabelEncoder()
df['educ'] = le.fit_transform(df['educ'])
df['race'] = le.fit_transform(df['race'])
df['gender'] = le.fit_transform(df['gender'])
df['income_cat'] = le.fit_transform(df['income_cat'])
df['voter_category'] = le.fit_transform(df['voter_category'])

df_mod = df[(df['q23_which_candidate_supporting'] == 1) | (df['q23_which_candidate_supporting'] == 2)]

# Remove target variable from features
# Also remove party identification questions since that is too obvious a tell
X = df_mod.drop(['q23_which_candidate_supporting', 'q30_partyidentification',
                 'q14_view_of_republicans', 'q15_view_of_democrats'], axis=1)
y = df_mod['q23_which_candidate_supporting']
X_train, y_train, X_test, y_test = train_test_split(X, y)

# Fit the random forest
# Get feature
rf = RandomForestClassifier()
cv = cross_validate(rf, X, y, cv=10)
print(cv)

# Get feature importances
rf2 = RandomForestClassifier()
rf2.fit(X=X, y=y, sample_weight=None)
feat_imp = list(zip(rf2.feature_importances_, X.columns))
print(sorted(feat_imp, reverse=True))