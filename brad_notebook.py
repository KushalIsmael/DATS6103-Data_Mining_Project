#------------------------------------------------------
# Import necessary packages
#------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------
# Read in csv file
#------------------------------------------------------

nv_df = pd.read_csv('nonvoters_data.csv')
# print(nv_df.shape)
# print(nv_df.columns)

#------------------------------------------------------
# Race Pie Chart
#------------------------------------------------------

distinct_races = set(nv_df['race'])
total_race = nv_df['race'].count()
hispanic_percentage = nv_df[nv_df['race'] == 'Hispanic']['race'].count()/total_race
other_mixed_percentage = nv_df[nv_df['race'] == 'Other/Mixed']['race'].count()/total_race
white_percentage = nv_df[nv_df['race'] == 'White']['race'].count()/total_race
black_percentage = nv_df[nv_df['race'] == 'Black']['race'].count()/total_race
race_percentages = [white_percentage, black_percentage, hispanic_percentage, other_mixed_percentage]
race_labels = ['White', 'Black', 'Hispanic', 'Other/Mixed']

race_pie, ax1 = plt.subplots()
ax1.pie(race_percentages, labels=race_labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(label = 'Percentage of Non-Voters by Race')
plt.show()

#------------------------------------------------------
# Gender Pie Chart
#------------------------------------------------------

distinct_genders = set(nv_df['gender'])
total_gender = nv_df['gender'].count()
male_percentage = nv_df[nv_df['gender'] == 'Male']['gender'].count()/total_gender
female_percentage = nv_df[nv_df['gender'] == 'Female']['gender'].count()/total_gender
gender_percentages = [male_percentage, female_percentage]
gender_labels = ['Male', 'Female']
gender_pie, ax1 = plt.subplots()
ax1.pie(gender_percentages, labels=gender_labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
plt.title(label = 'Percentage of Non-Voters by Gender')
plt.show()

#------------------------------------------------------
# Age Pie Chart
#------------------------------------------------------

age_labels_cut = ['twenties', 'thirties', 'forties', 'fifties', 'sixties', 'elderly']
age_bins= [20, 30, 40, 50, 60, 70, 200]
nv_df['Age_Group'] = pd.cut(nv_df['ppage'], bins = age_bins, labels = age_labels_cut, right = False)
total_age = nv_df['Age_Group'].count()
twenties = nv_df[nv_df['Age_Group'] == 'twenties']['Age_Group'].count()/total_age
thirties = nv_df[nv_df['Age_Group'] == 'thirties']['Age_Group'].count()/total_age
forties = nv_df[nv_df['Age_Group'] == 'forties']['Age_Group'].count()/total_age
fifties = nv_df[nv_df['Age_Group'] == 'fifties']['Age_Group'].count()/total_age
sixties = nv_df[nv_df['Age_Group'] == 'sixties']['Age_Group'].count()/total_age
elderly = nv_df[nv_df['Age_Group'] == 'elderly']['Age_Group'].count()/total_age
age_percentages = [twenties, thirties, forties, fifties, sixties, elderly]
age_labels = ['Twenties', 'Thirties', 'Forties', 'Fifties', 'Sixties', 'Elderly']
age_pie, ax1 = plt.subplots()
ax1.pie(age_percentages, labels=age_labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
plt.title(label = 'Percentage of Non-Voters by Age Group')
plt.show()

#------------------------------------------------------
# Education-Level Pie Chart
#------------------------------------------------------

distinct_educ = set(nv_df['educ'])
total_educ = nv_df['educ'].count()
hs_percentage = nv_df[nv_df['educ'] == 'High school or less']['educ'].count()/total_educ
some_college_percentage = nv_df[nv_df['educ'] == 'Some college']['educ'].count()/total_educ
college_percentage = nv_df[nv_df['educ'] == 'College']['educ'].count()/total_educ
educ_percentages = [hs_percentage, some_college_percentage, college_percentage]
educ_labels = ['High School or Less', 'Some College', 'College']

educ_pie, ax1 = plt.subplots()
ax1.pie(educ_percentages, labels=educ_labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
plt.title(label = 'Percentage of Non-Voters by Education-Level')
plt.show()

#------------------------------------------------------
# Income Category Pie Chart
#------------------------------------------------------

distinct_income = set(nv_df['income_cat'])
total_income= nv_df['income_cat'].count()

income1_percentage = nv_df[nv_df['income_cat'] == 'Less than $40k']['income_cat'].count()/total_income
income2_percentage = nv_df[nv_df['income_cat'] == '$40-75k']['income_cat'].count()/total_income
income3_percentage = nv_df[nv_df['income_cat'] == '$75-125k']['income_cat'].count()/total_income
income4_percentage = nv_df[nv_df['income_cat'] == '$125k or more']['income_cat'].count()/total_income
educ_percentages = [income1_percentage, income2_percentage, income3_percentage, income4_percentage]
educ_labels = ['Less than $40k', '$40-75k', '$75-125k', '$125k or more']

income_pie, ax1 = plt.subplots()
ax1.pie(educ_percentages, labels=educ_labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
plt.title(label = 'Percentage of Non-Voters by Education-Level')
plt.show()

#------------------------------------------------------
# Correlation Matrix
#------------------------------------------------------

corrMatrix = nv_df.corr()
sns.heatmap(corrMatrix)
plt.show()
