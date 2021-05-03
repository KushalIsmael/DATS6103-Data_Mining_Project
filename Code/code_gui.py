import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from pathlib import Path

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

        self.layout = QVBoxLayout(self.main_widget)

        self.questions = QPlainTextEdit(questions_text)
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
    # the methods for this class are:
    #   _init_
    #   initUi
    #   update
    #::---------------------------------------------------------

    def __init__(self):
        super(Demographics, self).__init__()
        self.Title = "Demographics of Survey Respondents"
        self.setWindowTitle(self.Title)

        # Race pie
        self.figure = Figure()
        self.ax1 = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)

        self.ax1.pie(race_percentages, labels=race_labels, autopct='%1.1f%%', startangle=90)
        self.ax1.axis('equal')
        self.ax1.set_title('Race')

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

        #Gender pie
        self.figure2 = Figure()
        self.ax2 = self.figure2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.figure2)

        self.ax2.pie(gender_percentages, labels=gender_labels, autopct='%1.1f%%', startangle=90)
        self.ax2.axis('equal')
        self.ax2.set_title('Gender')

        self.figure2.tight_layout()
        self.figure2.canvas.draw_idle()

        #Education pie
        self.figure3 = Figure()
        self.ax3 = self.figure3.add_subplot(111)
        self.canvas3 = FigureCanvas(self.figure3)

        self.ax3.pie(educ_percentages, labels=educ_labels, autopct='%1.1f%%', startangle=90)
        self.ax3.axis('equal')
        self.ax3.set_title('Education')

        self.figure3.tight_layout()
        self.figure3.canvas.draw_idle()

        #Income bar chart
        self.figure4 = Figure()
        self.ax4 = self.figure4.add_subplot(111)
        self.canvas4 = FigureCanvas(self.figure4)

        self.ax4.bar(income_labels, income_percentages, color=['tab:blue','tab:orange','tab:green','tab:red'])
        self.ax4.set_title('Income')

        self.figure4.tight_layout()
        self.figure4.canvas.draw_idle()

        #Age bar chart
        self.figure5 = Figure()
        self.ax5 = self.figure5.add_subplot(111)
        self.canvas5 = FigureCanvas(self.figure5)

        self.ax5.bar(age_labels, age_percentages, color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
        self.ax5.set_title('Age Groups')

        self.figure5.tight_layout()
        self.figure5.canvas.draw_idle()

        #Set layouts
        layout = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout3 = QHBoxLayout()

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

        #Questions option
        self.questionLabel = QLabel('Select a question')
        self.questionLabel.setFixedHeight(20)
        self.questionDrop = QComboBox()
        self.questionDrop.addItems(questions_list)
        self.questionDrop.setFixedHeight(20)

        #Demographics option
        self.demoLabel = QLabel('Select a demographic')
        self.demoLabel.setFixedHeight(20)
        self.demoDrop = QComboBox()
        self.demoDrop.setFixedHeight(20)
        self.demoDrop.addItems(demographics_list)

        #Plot button
        self.buttonPlot = QPushButton('Plot Distribution')
        self.buttonPlot.setFixedHeight(20)
        self.buttonPlot.clicked.connect(self.update)

        #Canvas for bar chart
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.axes = [self.ax]
        self.stackPlot = FigureCanvas(self.fig)

        #Set layouts
        self.layout.addWidget(self.questionLabel)
        self.layout.addWidget(self.questionDrop)
        self.layout.addWidget(self.demoLabel)
        self.layout.addWidget(self.demoDrop)
        self.layout.addWidget(self.buttonPlot)
        self.layout.addWidget(self.stackPlot)

        self.setCentralWidget(self.main_widget)
        self.resize(1000, 800)
        self.show()

    def update(self):
        self.ax.clear()

        #set values and create subset to chart
        demo = str(self.questionDrop.currentText())
        quest = str(self.demoDrop.currentText())
        dfpivot = dfo[[demo,quest]]

        #create pivot chart of selected data and plot
        pivot = dfpivot.pivot_table(index=quest, columns=demo, aggfunc=len, fill_value=0)
        pivot.plot(kind='bar', stacked=True, rot=0 , ax=self.ax, ylabel='Response Count')

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

class RandomForest(QMainWindow):

    #;:-------------------------------------------------
    # This class runs the random forest model based on
    # testing percentage and features.
    # the methods for this class are:
    #   _init_
    #   initUi
    #   update
    #::--------------------------------------------------

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Model"

        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)

        #Set layouts
        self.layout = QVBoxLayout(self.main_widget)
        self.SelectionBox = QHBoxLayout()
        self.ModelBox1 = QHBoxLayout()
        self.ResultsLayout = QVBoxLayout()
        self.ModelBox2 = QHBoxLayout()

        #Test percent option
        self.labelPercentTest = QLabel('Percent to test:')
        self.boxPercentTest = QSpinBox(self)
        self.boxPercentTest.setRange(1, 99)
        self.boxPercentTest.setValue(25)

        #Feature option
        self.labelFeature = QLabel('Number of Features:')
        self.boxFeature = QSpinBox(self)
        self.boxFeature.setRange(1, 92)
        self.boxFeature.setValue(25)

        #Button to run model
        self.buttonRun = QPushButton("Run Model")
        self.buttonRun.clicked.connect(self.update)

        self.SelectionBox.addWidget(self.labelPercentTest)
        self.SelectionBox.addWidget(self.boxPercentTest)
        self.SelectionBox.addWidget(self.labelFeature)
        self.SelectionBox.addWidget(self.boxFeature)
        self.SelectionBox.addWidget(self.buttonRun)

        #Classification report, accuracy score, roc score
        self.boxResults = QLabel('Classification Report:')
        self.Results = QPlainTextEdit()
        self.Results.setFixedWidth(325)
        self.Accuracy = QLabel('Accuracy Score:')
        self.ROC = QLabel('ROC Score:')

        self.ResultsLayout.addWidget(self.boxResults)
        self.ResultsLayout.addWidget(self.Results)
        self.ResultsLayout.addWidget(self.Accuracy)
        self.ResultsLayout.addWidget(self.ROC)

        self.ModelBox1.addLayout(self.ResultsLayout)

        #Confusion matrix plot
        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes1 = [self.ax1]
        self.canvasCFM = FigureCanvas(self.fig1)

        self.ModelBox2.addWidget(self.canvasCFM)

        #ROC plot
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvasROC = FigureCanvas(self.fig2)

        self.ModelBox2.addWidget(self.canvasROC)

        #Feature Importance plot
        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvasFI = FigureCanvas(self.fig3)

        self.ModelBox1.addWidget(self.canvasFI)


        self.layout.addLayout(self.SelectionBox)
        self.layout.addLayout(self.ModelBox1)
        self.layout.addLayout(self.ModelBox2)

        self.setCentralWidget(self.main_widget)
        self.resize(1000, 800)
        self.show()

    def update(self):

        testperc = float(self.boxPercentTest.value())/100
        num_features = int(self.boxFeature.value())

        '''
        :param df: Dataframe of all observations (train and test) to build model.
        :param num_features: The number of features to include in the model (all variables except target).
        :param test_percent: Percent of data to use in test (i.e. 0.3 means 70% train, 30% test).
        :return: accuracy_score_value: The accuracy of the RF model with the parameters passed above. (TP + FN)/ (TP + FP + TN + FN)
        :return: conf: Confusion matrix of RF model. This classifies the true positives, false positives, true negatives, false negatives.
        :return: auc_graph: Graph of AUC (area under curve) of the RF model.
        :return: auc_score_value: AUC (area under curve) score. Random guessing is 0.5, and closer to 1 means smarter model.
        :return: feature_importance_plot: Importance of the num_features chosen. Higher importance means it greater reduces entropy in classification.
        '''

        self.Results.clear()
        self.Accuracy.clear()
        self.ROC.clear()
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Create empty variables to return if the user passes in invalid parameters
        auc_null = np.nan
        conf_null = np.zeros((2, 2), dtype=int)
        auc__null_graph = plt.plot()
        auc_score_null = np.nan
        feature_importance_plot_null = plt.plot()

        # There are only 92 features available
        if (num_features < 1) or (num_features > 92):
            return auc_null, conf_null, auc__null_graph, auc_score_null, feature_importance_plot_null
        # We cannot test on 0 or 100 percent of our data
        if (testperc < 0.01) or (testperc > 0.99):
            return auc_null, conf_null, auc__null_graph, auc_score_null, feature_importance_plot_null

        # Go through modeling steps in this function
        # Start with getting X, y, and train-test split
        Xpre = df_mod.drop(columns=['q23_which_candidate_supporting'], axis=1)
        ypre = df_mod['q23_which_candidate_supporting']

        X_pre_train, X_pre_test, y_pre_train, y_pre_test = train_test_split(Xpre, ypre, test_size=testperc,
                                                                            random_state=1918)

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testperc, random_state=1918)

        # Fit the model
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)

        # Output: classifcation report
        self.results = str(classification_report(y_test,y_pred))
        self.Results.appendPlainText(self.results)

        # Output: accuracy metrics
        self.accuracy_score_value = 'Accuracy: '+str(accuracy_score(y_test, y_pred) * 100)
        self.Accuracy.setText(self.accuracy_score_value)

        # Output: confusion matrix
        conf = confusion_matrix(y_test, y_pred)

        class_names = df_mod['q23_which_candidate_supporting'].unique()

        self.ax1.set_xlabel('Predicted')
        self.ax1.set_ylabel('Actual')
        self.ax1.set_yticks((0, 1))
        self.ax1.set_xticks((0, 1))
        self.ax1.set_yticklabels(class_names)
        self.ax1.set_xticklabels(class_names)
        self.ax1.set_title('Confusion Matrix')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                self.ax1.text(j, i, str(conf[i][j]))
        self.ax1.matshow(conf, cmap= plt.cm.get_cmap('Blues', 14))

        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        # Output: ROC Curve
        fpr, tpr, threholds = roc_curve(y_test.values, y_pred_proba[:, 1], pos_label=2)
        self.ax2.plot(fpr,tpr)
        self.ax2.set_title("ROC Curve")
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        # Output: ROC score
        self.roc_score_value = 'ROC Score: ' + str(roc_auc_score(y_test, y_pred_proba[:, 1])*100)
        self.ROC.setText(self.roc_score_value)


        # Output Feature importance
        imp_final = rf.feature_importances_
        feat_imp_final = pd.Series(imp_final, X_train.columns)
        feat_imp_final.sort_values(ascending=True, inplace=True)
        self.ax3.barh(feat_imp_final.index,feat_imp_final.values)
        self.ax3.set_yticks(np.arange(len(feat_imp_final.index)))
        self.ax3.set_yticklabels(feat_imp_final.index)
        self.ax3.set_aspect('auto')
        self.ax3.set_title('Feature Importance')

        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()


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
        self.setWindowIcon(QIcon('Code//Icons//exit.png')) #Icon made by Pixel perfect from www.flaticon.com
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

        exitButton = QAction(QIcon('exit.png'), '&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::--------------------------------------
        # EDA action
        # Create Survey, Demographics, Distributions options
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

        distributionButton = QAction(QIcon('insight.png'),'Distribution Charts', self)
        distributionButton.setStatusTip('Bar charts of demographics by question')
        distributionButton.triggered.connect(self.edaDistributions)
        edaMenu.addAction(distributionButton)

        #::--------------------------------------
        # Model Action
        # Create Random Forest option
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

    global dfo
    global df_mod
    global demographics_data
    global demographics_list
    global questions_text
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
    df = pd.read_csv(dir + '\\' + 'nonvoters_data.csv', sep=',', header=0)

    # If having issues loading in, then run this:
    # df = pd.read_csv('nonvoters_data.csv')

    ##### Exploratory data analysis ##### ---------------------------------------------------------------------------------

    initial_cols = df.columns

    #### Data Pre-Processing to prepare for modeling ##### -----------------------------------------------------------------

    # Rename columns to descriptive names
    df.columns = ['RespId', 'weight',
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
                  'Age', 'Education', 'Race', 'Gender', 'Income', 'voter_category'
                  ]

    # Drop irrelevant fields (US Citizen, responder ID, observation weight)
    # Drop questions that were not asked to all participants (i.e. "why did you vote" to non-voters, "Republican type" for Democrats)
    df.drop(['q1_uscitizen', 'q22_whynotvoting_2020',
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

    # Step 1 - Add column, Age_Group
    age_labels_cut = ['twenties', 'thirties', 'forties', 'fifties', 'sixties', 'seventies +']
    age_bins = [20, 30, 40, 50, 60, 70, 200]
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels_cut, right=False)

    dfo = df.copy()

    # Step 2 - Replace -1 or -1.0 values with NaN
    # Values might be stored as int or float, so account for both
    df[replace_neg_one] = df[replace_neg_one].replace(-1, np.nan)
    df[replace_neg_one] = df[replace_neg_one].replace(-1.0, np.nan)

    # Step 3 - Replace NaN with demographic mean
    for x in replace_neg_one:
        df[x] = df[x].fillna(df.groupby(by=['Education', 'Race', 'Gender', 'Income'])[x].transform('mean'))

    total_age = df['Age_Group'].count()
    twenties = df[df['Age_Group'] == 'twenties']['Age_Group'].count() / total_age
    thirties = df[df['Age_Group'] == 'thirties']['Age_Group'].count() / total_age
    forties = df[df['Age_Group'] == 'forties']['Age_Group'].count() / total_age
    fifties = df[df['Age_Group'] == 'fifties']['Age_Group'].count() / total_age
    sixties = df[df['Age_Group'] == 'sixties']['Age_Group'].count() / total_age
    elderly = df[df['Age_Group'] == 'seventies +']['Age_Group'].count() / total_age
    age_percentages = [twenties, thirties, forties, fifties, sixties, elderly]
    age_labels = ['Twenties', 'Thirties', 'Forties', 'Fifties', 'Sixties', 'Seventies +']



    distinct_races = set(df['Race'])
    total_race = df['Race'].count()
    hispanic_percentage = df[df['Race'] == 'Hispanic']['Race'].count() / total_race
    other_mixed_percentage = df[df['Race'] == 'Other/Mixed']['Race'].count() / total_race
    white_percentage = df[df['Race'] == 'White']['Race'].count() / total_race
    black_percentage = df[df['Race'] == 'Black']['Race'].count() / total_race
    race_percentages = [white_percentage, black_percentage, hispanic_percentage, other_mixed_percentage]
    race_labels = ['White', 'Black', 'Hispanic', 'Other/Mixed']

    distinct_genders = set(df['Gender'])
    total_gender = df['Gender'].count()
    male_percentage = df[df['Gender'] == 'Male']['Gender'].count() / total_gender
    female_percentage = df[df['Gender'] == 'Female']['Gender'].count() / total_gender
    gender_percentages = [male_percentage, female_percentage]
    gender_labels = ['Male', 'Female']

    distinct_educ = set(df['Education'])
    total_educ = df['Education'].count()
    hs_percentage = df[df['Education'] == 'High school or less']['Education'].count() / total_educ
    some_college_percentage = df[df['Education'] == 'Some college']['Education'].count() / total_educ
    college_percentage = df[df['Education'] == 'College']['Education'].count() / total_educ
    educ_percentages = [hs_percentage, some_college_percentage, college_percentage]
    educ_labels = ['High School or Less', 'Some College', 'College']

    distinct_income = set(df['Income'])
    total_income = df['Income'].count()

    income1_percentage = df[df['Income'] == 'Less than $40k']['Income'].count() / total_income
    income2_percentage = df[df['Income'] == '$40-75k']['Income'].count() / total_income
    income3_percentage = df[df['Income'] == '$75-125k']['Income'].count() / total_income
    income4_percentage = df[df['Income'] == '$125k or more']['Income'].count() / total_income
    income_percentages = [income1_percentage, income2_percentage, income3_percentage, income4_percentage]
    income_labels = ['Less than $40k', '$40-75k', '$75-125k', '$125k or more']

    # Set demographics variable
    demographics_data = df[['Age_Group','Education', 'Race', 'Gender', 'Income']]
    demographics_list = list(demographics_data.columns)

    # Set questions variable
    questions = df.iloc[:,0:87]
    questions_list = list(questions.columns)
    questions_text = '\n'.join(map(str, questions_list))

    le = LabelEncoder()
    df['Education'] = le.fit_transform(df['Education'])
    df['Race'] = le.fit_transform(df['Race'])
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Income'] = le.fit_transform(df['Income'])
    df['voter_category'] = le.fit_transform(df['voter_category'])
    df['Age_Group'] = le.fit_transform(df['Age_Group'])

    # For q23_which_candidate_supporting, value of 1 is Trump and value of 2 is Biden
    # Drop unsure (value of 3) and refused to answer (value of -1) to set up our two-way classification
    df_mod = df[(df['q23_which_candidate_supporting'] == 1) | (df['q23_which_candidate_supporting'] == 2)]

if __name__ == '__main__':
    #run voter_turnout to read in data before running GUI app
    voter_turnout()
    main()



