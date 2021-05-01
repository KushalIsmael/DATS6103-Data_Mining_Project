import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys


from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class Survey(QMainWindow):

    #;:------------------------------------------------------
    # This class displays the list of questions in the survey
    #   _init_
    #   initUi
    #::------------------------------------------------------

    def __init__(self):
        super().__init__()

    def __init__(self):
        super(Survey, self).__init__()

        self.Title = "Survey Questions"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget) #set vertical layout

        self.questions = QLabel('1 - Are you a US Citizen? 1.Yes/2.No'
                                '\n2 - In your view, how important are each of the following to being a good American?'
                                '\n -Voting in elections 1.Very Important/2.Somewhat Import/3.Not so important/4.Not at all important')
                                #set text for questions widget

        self.layout.addWidget(self.questions)

        self.setCentralWidget(self.main_widget)
        self.resize(1000, 800)
        self.show()

class Canvas(FigureCanvas):

    #;:--------------------------------------------------------
    # This class creates a standard canvas size to draw charts
    #::--------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class Demographics(QDialog):

    #;:---------------------------------------------------------
    # This class creates draws pie charts and bar charts to show
    # the distribution of the demographics of survey respondents
    #::---------------------------------------------------------

    def __init__(self):
        super(Demographics, self).__init__()
        self.Title = "Demographics of Survey Respondents"

        # Race pie
        self.figure = Figure()
        self.ax1 = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)

        self.ax1.pie(race_percentages, labels=race_labels, autopct='%1.1f%%', startangle=90)
        self.ax1.axis('equal')

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

        #Gender pie
        self.figure2 = Figure()
        self.ax2 = self.figure2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.figure2)

        self.ax2.pie(gender_percentages, labels=gender_labels, autopct='%1.1f%%', startangle=90)
        self.ax2.axis('equal')

        self.figure2.tight_layout()
        self.figure2.canvas.draw_idle()

        #Education pie
        self.figure3 = Figure()
        self.ax3 = self.figure3.add_subplot(111)
        self.canvas3 = FigureCanvas(self.figure3)

        self.ax3.pie(educ_percentages, labels=educ_labels, autopct='%1.1f%%', startangle=90)
        self.ax3.axis('equal')

        self.figure3.tight_layout()
        self.figure3.canvas.draw_idle()

        #Income bar chart
        self.figure4 = Figure()
        self.ax4 = self.figure4.add_subplot(111)
        self.canvas4 = FigureCanvas(self.figure4)

        self.ax4.bar(income_labels, income_percentages, color=['tab:blue','tab:orange','tab:green','tab:red'])

        self.figure4.tight_layout()
        self.figure4.canvas.draw_idle()

        #Age bar chart
        self.figure5 = Figure()
        self.ax5 = self.figure5.add_subplot(111)
        self.canvas5 = FigureCanvas(self.figure5)

        self.ax5.bar(age_labels, age_percentages, color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])

        self.figure5.tight_layout()
        self.figure5.canvas.draw_idle()


        layout = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout3 = QHBoxLayout()

        # adding canvas to the layout
        layout2.addWidget(self.canvas)
        layout2.addWidget(self.canvas2)
        layout2.addWidget(self.canvas3)
        layout.addLayout(layout2)

        layout3.addWidget(self.canvas4)
        layout3.addWidget(self.canvas5)
        layout.addLayout(layout3)

        # setting layout to the main window
        self.setLayout(layout)

class Distributions(QMainWindow):
    #;:-------------------------------------------------------------
    # This class creates a canvas to draw a stacked bar chart
    # It presents option to choose demographic feature and question
    # the methods for this class are:
    #   _init_
    #   initUi
    #   update
    #::-------------------------------------------------------------
    #send_fig = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def __init__(self):
        super(Distributions, self).__init__()

        self.Title = "Distribution of Demographic for Questions"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.questionLabel = QLabel('Select a question')
        self.questionDrop = QComboBox()
        self.questionDrop.addItems(questions_list)

        self.demoLabel = QLabel('Select a demographic')
        self.demoDrop = QComboBox()
        self.demoDrop.addItems(demographics_list)

        self.buttonPlot = QPushButton('Plot Distribution')
        #self.buttonPlot.clicked.connect(self.update)

        self.stackPlot = QLabel('Stacked bar chart of plot')

        self.layout.addWidget(self.questionLabel)
        self.layout.addWidget(self.questionDrop)
        self.layout.addWidget(self.demoLabel)
        self.layout.addWidget(self.demoDrop)
        self.layout.addWidget(self.buttonPlot)
        self.layout.addWidget(self.stackPlot)

        self.setCentralWidget(self.main_widget)
        self.resize(1000, 800)
        self.show()

class RandomForest(QMainWindow):

    #;:-------------------------------------------------
    # This class runs the random forest model based on
    # testing percentage and features.
    # the methods for this class are:
    #   _init_
    #   initUi
    #   update
    #::--------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Model"

        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)
        self.SelectionBox = QHBoxLayout()  # sets horizontal layout for selection box
        self.ModelBox1 = QHBoxLayout()  # sets horizontal layout for Results and Conf Matrix
        self.Results = QVBoxLayout()  # set vertical layout for Results and Accuracy score
        self.ModelBox2 = QHBoxLayout()  # sets horizontal layout for Feature Importance and ROC

        self.labelPercentTest = QLabel('Percent to test:')
        self.boxPercentTest = QSpinBox(self)
        self.boxPercentTest.setRange(1, 100)
        self.boxPercentTest.setValue(30)

        self.labelFeature = QLabel('Top Features:')
        self.boxFeature = QSpinBox(self)
        self.boxFeature.setRange(1, 20)
        self.boxFeature.setValue(20)

        self.buttonRun = QPushButton("Run Model")
        # self.buttonRun.clicked.connect(self.update) update function not yet set

        self.SelectionBox.addWidget(self.labelPercentTest)
        self.SelectionBox.addWidget(self.boxPercentTest)
        self.SelectionBox.addWidget(self.labelFeature)
        self.SelectionBox.addWidget(self.boxFeature)
        self.SelectionBox.addWidget(self.buttonRun)


        self.boxResults = QLabel('Results Output from Model')
        self.boxAccuracy = QLabel('Accuracy Score')

        self.Results.addWidget(self.boxResults)
        self.Results.addWidget(self.boxAccuracy)

        self.ModelBox1.addLayout(self.Results)

        self.boxMatrix = QLabel('Confusion Matrix')
        self.ModelBox1.addWidget(self.boxMatrix)


        self.boxFeatImportance = QLabel('Feature Importance Graph')
        self.boxROC = QLabel('ROC Curve')

        self.ModelBox2.addWidget(self.boxFeatImportance)
        self.ModelBox2.addWidget(self.boxROC)


        self.layout.addLayout(self.SelectionBox)
        self.layout.addLayout(self.ModelBox1)
        self.layout.addLayout(self.ModelBox2)

        self.setCentralWidget(self.main_widget)
        self.resize(1000, 800)
        self.show()

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

        self.setWindowIcon(QIcon('vote.png')) #Icon made by Pixel perfect from www.flaticon.com
        self.Title = 'Voter Turnout Survey'

        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the menu and items
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

        exitButton = QAction(QIcon('Icons//exit.png'), '&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::--------------------------------------
        # EDA action
        # Create Survey, Demographics, Distributions options with icons
        # Text tips for each option
        #::--------------------------------------
        questionsButton = QAction(QIcon('question.png'),'Survey', self)
        questionsButton.setStatusTip('Questions in survey')
        questionsButton.triggered.connect(self.edaSurvey)
        edaMenu.addAction(questionsButton)

        demographicsButton = QAction(QIcon('pie-chart.png'),'Demographics', self)
        demographicsButton.setStatusTip('Graphs of demographics in survey')
        demographicsButton.triggered.connect(self.edaDemographics)
        edaMenu.addAction(demographicsButton)

        distributionButton = QAction(QIcon('bar-chart.png'),'Distribution Charts', self)
        distributionButton.setStatusTip('Bar charts of demographics by question')
        distributionButton.triggered.connect(self.edaDistributions)
        edaMenu.addAction(distributionButton)

        #::--------------------------------------
        # Model Action
        # Create Random Forest option with icon
        # Random forest text tip
        #::--------------------------------------

        randomforestButton = QAction(QIcon('forest.png'),'Random Forest', self)
        randomforestButton.setStatusTip('Run a random forest model on the data')
        randomforestButton.triggered.connect(self.modelRF)
        modelMenu.addAction(randomforestButton)

        self.dialogs = list()
        self.setStatusBar(QStatusBar(self))

    #::--------------------------------------
    # Connect the button actions to classes
    #::--------------------------------------

    def edaSurvey(self):
        dialog = Survey()
        self.dialogs.append(dialog)
        dialog.show()

    def edaDemographics(self):
        dialog = Demographics()
        self.dialogs.append(dialog)
        dialog.show()

    def edaDistributions(self):
        dialog = Distributions()
        self.dialogs.append(dialog)
        dialog.show()

    def modelRF(self):
        dialog = RandomForest()
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

def voter_turnout():
    # read in data and set global variables to be used in models and graphs

    global demographics_data
    global demographics_list
    global questions
    global questions_list
    global race_percentages
    global race_labels
    global educ_labels
    global educ_percentages
    global gender_labels
    global gender_percentages
    global income_labels
    global income_percentages
    global age_labels
    global age_percentages

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
                  'age', 'Education', 'Race', 'Gender', 'Income', 'voter_category'
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
        nv_df[x] = nv_df[x].fillna(nv_df.groupby(by=['age', 'Education', 'Race', 'Gender', 'Income'])[x].transform('mean'))

    # Create age bins
    age_labels_cut = ['twenties', 'thirties', 'forties', 'fifties', 'sixties', 'seventies +']
    age_bins = [20, 30, 40, 50, 60, 70, 200]
    nv_df['Age Group'] = pd.cut(nv_df['age'], bins=age_bins, labels=age_labels_cut, right=False)

    total_age = nv_df['Age Group'].count()
    twenties = nv_df[nv_df['Age Group'] == 'twenties']['Age Group'].count() / total_age
    thirties = nv_df[nv_df['Age Group'] == 'thirties']['Age Group'].count() / total_age
    forties = nv_df[nv_df['Age Group'] == 'forties']['Age Group'].count() / total_age
    fifties = nv_df[nv_df['Age Group'] == 'fifties']['Age Group'].count() / total_age
    sixties = nv_df[nv_df['Age Group'] == 'sixties']['Age Group'].count() / total_age
    elderly = nv_df[nv_df['Age Group'] == 'seventies +']['Age Group'].count() / total_age
    age_percentages = [twenties, thirties, forties, fifties, sixties, elderly]
    age_labels = ['Twenties', 'Thirties', 'Forties', 'Fifties', 'Sixties', 'Seventies +']



    distinct_races = set(nv_df['Race'])
    total_race = nv_df['Race'].count()
    hispanic_percentage = nv_df[nv_df['Race'] == 'Hispanic']['Race'].count() / total_race
    other_mixed_percentage = nv_df[nv_df['Race'] == 'Other/Mixed']['Race'].count() / total_race
    white_percentage = nv_df[nv_df['Race'] == 'White']['Race'].count() / total_race
    black_percentage = nv_df[nv_df['Race'] == 'Black']['Race'].count() / total_race
    race_percentages = [white_percentage, black_percentage, hispanic_percentage, other_mixed_percentage]
    race_labels = ['White', 'Black', 'Hispanic', 'Other/Mixed']

    distinct_genders = set(nv_df['Gender'])
    total_gender = nv_df['Gender'].count()
    male_percentage = nv_df[nv_df['Gender'] == 'Male']['Gender'].count() / total_gender
    female_percentage = nv_df[nv_df['Gender'] == 'Female']['Gender'].count() / total_gender
    gender_percentages = [male_percentage, female_percentage]
    gender_labels = ['Male', 'Female']

    distinct_educ = set(nv_df['Education'])
    total_educ = nv_df['Education'].count()
    hs_percentage = nv_df[nv_df['Education'] == 'High school or less']['Education'].count() / total_educ
    some_college_percentage = nv_df[nv_df['Education'] == 'Some college']['Education'].count() / total_educ
    college_percentage = nv_df[nv_df['Education'] == 'College']['Education'].count() / total_educ
    educ_percentages = [hs_percentage, some_college_percentage, college_percentage]
    educ_labels = ['High School or Less', 'Some College', 'College']

    distinct_income = set(nv_df['Income'])
    total_income = nv_df['Income'].count()

    income1_percentage = nv_df[nv_df['Income'] == 'Less than $40k']['Income'].count() / total_income
    income2_percentage = nv_df[nv_df['Income'] == '$40-75k']['Income'].count() / total_income
    income3_percentage = nv_df[nv_df['Income'] == '$75-125k']['Income'].count() / total_income
    income4_percentage = nv_df[nv_df['Income'] == '$125k or more']['Income'].count() / total_income
    income_percentages = [income1_percentage, income2_percentage, income3_percentage, income4_percentage]
    income_labels = ['Less than $40k', '$40-75k', '$75-125k', '$125k or more']

    # Set demographics variable
    demographics_data = nv_df[['Age Group','Education', 'Race', 'Gender', 'Income']]
    demographics_list = list(demographics_data.columns)

    # Set questions variable
    questions = nv_df.iloc[:,0:87]
    questions_list = list(questions.columns)

if __name__ == '__main__':
    voter_turnout()
    main()



