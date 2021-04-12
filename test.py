#------------------------------------------------------
# Import necessary packages
#------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------
# Read in csv file
#------------------------------------------------------

nv_df = pd.read_csv('nonvoters_data.csv')
#print(nv_df.shape)
print(nv_df.columns)
#print(nv_df['ppage'].sort_values())
#print(nv_df['ppage'].sort_values(ascending=False))
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
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
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
