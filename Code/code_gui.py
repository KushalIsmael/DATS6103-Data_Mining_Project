import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys


from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, \
                            QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Figure
import matplotlib.pyplot as plt




#gui
#todo create class for EDA
class eda(QMainWindow):
    def __init__(self):
        super().__init__()

#can use similar code to demo.py for Random Forest elements
#todo create class for Random Forest
    #todo create box for ROC
    #todo create box for feature preference
    #todo create box for correlation matrix
    #todo create box for bar chart plot

class BarChartPlot(QWidget):
    #;:-----------------------------------------------------------------------
    # This class creates a canvas to draw a stacked bar chart
    # It presents option to choose demographic feature and question
    # the methods for this class are:
    #   _init_
    #   initUi
    #   update
    #::-------------
    #send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout to draw a bar chart
        # The layout sets all the elements and manage the changes
        # made on the canvas
        #::--------------------------------------------------------

        super(self).__init__()
        #create widget title
        self.Title = "Distribution of Responses by Demographic"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        #create widget layout as vertical box
        layout = QVBoxLayout()
        self.setLayout(layout)
        #add combo box 1

        #add combo box 2

        #set plot size
        self.canvas = Figure(plt.Figure(dpi=100, figsize=(800,800)))
        layout.addWidget(self.canvas)
        #add plot axes
        self.insert_ax()

    def insert_ax(self):
        self.ax = self.canvas.figure.subplots()
        self.ax.set_ylim([0, 5836])
        self.ax.set_xlim([0,5])
        self.bar = None

    def update_chart(self):
        labels = ['G1', 'G2', 'G3', 'G4', 'G5']
        men_means = [20, 35, 30, 35, 27]
        women_means = [25, 32, 34, 20, 25]
        men_std = [2, 3, 4, 1, 2]
        women_std = [3, 5, 2, 3, 3]
        width = 0.35  # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots()

        ax.bar(labels, men_means, width, yerr=men_std, label='Men')
        ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
               label='Women')

        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.legend()

        plt.show()

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        #::-----------------------
        #:: size of gui window
        #::-----------------------
        self.left = 100
        self.top = 100
        self.width = 1000
        self.height = 800

        #:: Title for the application

        self.Title = 'Voter Turnout Survey'

        #:: The initUi is call to create all the necessary elements for the menu

        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the menu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Menu bar items
        #::-----------------------------
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        edaMenu = mainMenu.addMenu('EDA')
        modelMenu = mainMenu.addMenu('Model')

        #::--------------------------------------
        # Exit action
        # Create exit option
        # Exit keyboard shortcut
        # Exit text tip
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), '&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::--------------------------------------
        # EDA action
        # Create exit option
        # Exit keyboard shortcut
        # Exit text tip
        #::--------------------------------------

        distributionButton = QAction('Distribution Charts', self)
        distributionButton.setStatusTip('Bar charts of demographics by question')
        #distributionButton.triggered(self.distribution)
        edaMenu.addAction(distributionButton)

        randomforestButton = QAction('Random Forest', self)
        randomforestButton.setStatusTip('Random forest model')
        modelMenu.addAction(randomforestButton)

    def distribution(self):
        dialog = BarChartPlot()
        self.dialogs.append(dialog)
        dialog.show()




#::------------------------
#:: Application starts here
#::------------------------
def main():
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = App()  # Creates the menu
    mn.show()
    sys.exit(app.exec_())  # Close the application

#todo create def to read in data and assign global variables
def voter_turnout():
    # read in data and set global variables to be used in models and graphs
    global demographics

    dir = str(Path(os.getcwd()).parents[0])
    nv_df = pd.read_csv(dir + '\\' + 'nonvoters_data.csv', sep=',', header=0)

    nv_df.columns = ['RespId', 'weight',
                  'q1_uscitizen',
                  'q2_important_voting', 'q2_important_jury', 'q2_important_following', 'q2_important_displaying',
                  'q2_important_census',
                  'q2_important_pledge', 'q2_important_military', 'q2_important_respect', 'q2_important_god',
                  'q2_important_protesting',
                  'q3_statement_racism1', 'q3_statement_racism2', 'q3_statement_feminine',
                  'q3_statement_msm', 'q3_statement_politiciansdontcare', 'q3_statement_besensitive',
                  'q4_impact_officialsfed', 'q4_impact_officialsstate', 'q4_impact_officialslocal',
                  'q4_impact_news', 'q4_impact_wallstreet', 'q4_impact_lawenforcement',
                  'q5_electionmatters',
                  'q6_officialsarelikeyou',
                  'q7_governmentdesign',
                  'q8_trust_presidency', 'q8_trust_congress', 'q8_trust_supremecourt', 'q8_trust_cdc',
                  'q8_trust_electedofficials',
                  'q8_trust_fbicia', 'q8_trust_newsmedia', 'q8_trust_police', 'q8_trust_postalservice',
                  'q9_politicalsystems_democracy', 'q9_politicalsystems_experts', 'q9_politicalsystems_strongleader',
                  'q9_politicalsystems_army',
                  'q10_disability', 'q10_chronic_illness', 'q10_unemployed', 'q10_evicted',
                  'q11_lostjob', 'q11_gotcovid', 'q11_familycovid',
                  'q11_coviddeath', 'q11_worriedmoney', 'q11_quitjob',
                  'q14_view_of_republicans',
                  'q15_view_of_democrats',
                  'q16_how_easy_vote',
                  'q17_secure_votingmachines', 'q17_secure_paperballotsinperson', 'q17_secure_paperballotsmail',
                  'q17_secure_electronicvotesonline',
                  'q18_votingsituations1', 'q18_votingsituations2', 'q18_votingsituations3', 'q18_votingsituations4',
                  'q18_votingsituations5',
                  'q18_votingsituations6', 'q18_votingsituations7', 'q18_votingsituations8', 'q18_votingsituations9',
                  'q18_votingsituations10',
                  'q19_get_more_voting1', 'q19_get_more_voting2', 'q19_get_more_voting3', 'q19_get_more_voting4',
                  'q19_get_more_voting5',
                  'q19_get_more_voting6', 'q19_get_more_voting7', 'q19_get_more_voting8', 'q19_get_more_voting9',
                  'q19_get_more_voting10',
                  'q20_currentlyregistered',
                  'q21_plan_to_vote',
                  'q22_whynotvoting_2020',
                  'q23_which_candidate_supporting',
                  'q24_preferred_voting_method',
                  'q25_howcloselyfollowing_election',
                  'q26_which_voting_category',
                  'q27_didyouvotein18', 'q27_didyouvotein16', 'q27_didyouvotein14',
                  'q27_didyouvotein12', 'q27_didyouvotein10', 'q27_didyouvotein08',
                  'q28_whydidyouvote_past1', 'q28_whydidyouvote_past2', 'q28_whydidyouvote_past3',
                  'q28_whydidyouvote_past4',
                  'q28_whydidyouvote_past5', 'q28_whydidyouvote_past6', 'q28_whydidyouvote_past7',
                  'q28_whydidyouvote_past8',
                  'q29_whydidyounotvote_past1', 'q29_whydidyounotvote_past2', 'q29_whydidyounotvote_past3',
                  'q29_whydidyounotvote_past4', 'q29_whydidyounotvote_past5',
                  'q29_whydidyounotvote_past6', 'q29_whydidyounotvote_past7', 'q29_whydidyounotvote_past8',
                  'q29_whydidyounotvote_past9', 'q29_whydidyounotvote_past10',
                  'q30_partyidentification',
                  'q31_republicantype',
                  'q32_democratictype',
                  'q33_closertowhichparty',
                  'age', 'educ', 'race', 'gender', 'income_cat', 'voter_category'
                  ]

    # Drop irrelevant fields (US Citizen, responder ID, observation weight)
    # Drop questions that were not asked to all participants (i.e. "why did you vote" to non-voters, "Republican type" for Democrats)
    nv_df.drop(['q1_uscitizen', 'q22_whynotvoting_2020',
             'q28_whydidyouvote_past1', 'q28_whydidyouvote_past2', 'q28_whydidyouvote_past3', 'q28_whydidyouvote_past4',
             'q28_whydidyouvote_past5', 'q28_whydidyouvote_past6', 'q28_whydidyouvote_past7', 'q28_whydidyouvote_past8',
             'q29_whydidyounotvote_past1', 'q29_whydidyounotvote_past2', 'q29_whydidyounotvote_past3',
             'q29_whydidyounotvote_past4', 'q29_whydidyounotvote_past5',
             'q29_whydidyounotvote_past6', 'q29_whydidyounotvote_past7', 'q29_whydidyounotvote_past8',
             'q29_whydidyounotvote_past9', 'q29_whydidyounotvote_past10',
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
        'q2_important_voting', 'q2_important_jury', 'q2_important_following', 'q2_important_displaying',
        'q2_important_census',
        'q2_important_pledge', 'q2_important_military', 'q2_important_respect', 'q2_important_god',
        'q2_important_protesting',
        'q3_statement_racism1', 'q3_statement_racism2', 'q3_statement_feminine',
        'q3_statement_msm', 'q3_statement_politiciansdontcare', 'q3_statement_besensitive',
        'q4_impact_officialsfed', 'q4_impact_officialsstate', 'q4_impact_officialslocal',
        'q4_impact_news', 'q4_impact_wallstreet', 'q4_impact_lawenforcement',
        'q5_electionmatters',
        'q6_officialsarelikeyou',
        'q7_governmentdesign',
        'q8_trust_presidency', 'q8_trust_congress', 'q8_trust_supremecourt', 'q8_trust_cdc',
        'q8_trust_electedofficials',
        'q8_trust_fbicia', 'q8_trust_newsmedia', 'q8_trust_police', 'q8_trust_postalservice',
        'q9_politicalsystems_democracy', 'q9_politicalsystems_experts', 'q9_politicalsystems_strongleader',
        'q9_politicalsystems_army',
        'q10_disability', 'q10_chronic_illness', 'q10_unemployed', 'q10_evicted',
        'q11_lostjob', 'q11_gotcovid', 'q11_familycovid',
        'q11_coviddeath', 'q11_worriedmoney', 'q11_quitjob',
        'q14_view_of_republicans',
        'q15_view_of_democrats',
        'q16_how_easy_vote',
        'q17_secure_votingmachines', 'q17_secure_paperballotsinperson', 'q17_secure_paperballotsmail',
        'q17_secure_electronicvotesonline',
        'q18_votingsituations1', 'q18_votingsituations2', 'q18_votingsituations3', 'q18_votingsituations4',
        'q18_votingsituations5',
        'q18_votingsituations6', 'q18_votingsituations7', 'q18_votingsituations8', 'q18_votingsituations9',
        'q18_votingsituations10',
        'q20_currentlyregistered',
        'q24_preferred_voting_method',
        'q25_howcloselyfollowing_election',
        'q26_which_voting_category',
        'q27_didyouvotein18', 'q27_didyouvotein16', 'q27_didyouvotein14',
        'q27_didyouvotein12', 'q27_didyouvotein10', 'q27_didyouvotein08',
        'q30_partyidentification'
    ]

    # Step 1 - Replace -1 or -1.0 values with NaN
    # Values might be stored as int or float, so account for both
    nv_df[replace_neg_one] = nv_df[replace_neg_one].replace(-1, np.nan)
    nv_df[replace_neg_one] = nv_df[replace_neg_one].replace(-1.0, np.nan)

    # Step 2 - Replace NaN with demographic mean
    for x in replace_neg_one:
        nv_df[x] = nv_df[x].fillna(nv_df.groupby(by=['age', 'educ', 'race', 'gender', 'income_cat'])[x].transform('mean'))

    # Create age bins
    age_labels_cut = ['twenties', 'thirties', 'forties', 'fifties', 'sixties', 'elderly']
    age_bins = [20, 30, 40, 50, 60, 70, 200]
    nv_df['Age_Group'] = pd.cut(nv_df['age'], bins=age_bins, labels=age_labels_cut, right=False)

    # Set demographics variable
    demographics = nv_df['Age_Group', 'educ', 'race', 'gender', 'income_cat','voter_category']

if __name__ == '__main__':
    main()



