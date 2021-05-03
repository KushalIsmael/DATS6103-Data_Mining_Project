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
        self.setWindowIcon(QIcon('Icons//question.png'))
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
        self.setWindowIcon(QIcon('Icons//pie-chart.png'))
        self.setWindowTitle(self.Title)

        # Race pie
        self.figure = Figure()
        self.ax1 = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)


        self.ax1.set_title('Race')

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

        #Gender pie
        self.figure2 = Figure()
        self.ax2 = self.figure2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.figure2)


        self.ax2.set_title('Gender')

        self.figure2.tight_layout()
        self.figure2.canvas.draw_idle()

        #Education pie
        self.figure3 = Figure()
        self.ax3 = self.figure3.add_subplot(111)
        self.canvas3 = FigureCanvas(self.figure3)


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
        self.setWindowIcon(QIcon('Icons//insight.png'))
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
        self.setWindowIcon(QIcon('Icons//forest.png'))

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

        self.Results.clear()
        self.Accuracy.clear()
        self.ROC.clear()
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Output: classifcation report
        self.results = str(classification_report(y_test,y_pred))
        self.Results.appendPlainText(self.results)

        # Output: accuracy metrics
        self.accuracy_score_value = 'Accuracy: '+str(accuracy_score(y_test, y_pred) * 100)
        self.Accuracy.setText(self.accuracy_score_value)


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


        self.roc_score_value = 'ROC Score: ' + str(roc_auc_score(y_test, y_pred_proba[:, 1])*100)
        self.ROC.setText(self.roc_score_value)



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
        self.setWindowIcon(QIcon('Icons//vote.png')) #Icon made by Pixel perfect from www.flaticon.com
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
        # Create Survey, Demographics, Distributions options
        # Text tips for each option
        #::--------------------------------------
        questionsButton = QAction(QIcon('Icons//question.png'),'Survey', self)
        questionsButton.setStatusTip('Questions in survey')
        questionsButton.triggered.connect(self.edaSurvey)
        edaMenu.addAction(questionsButton)

        demographicsButton = QAction(QIcon('Icons//pie-chart.png'),'Demographics', self)
        demographicsButton.setStatusTip('Graphs of demographics in survey')
        demographicsButton.triggered.connect(self.edaDemographics)
        edaMenu.addAction(demographicsButton)

        distributionButton = QAction(QIcon('Icons//insight.png'),'Distribution Charts', self)
        distributionButton.setStatusTip('Bar charts of demographics by question')
        distributionButton.triggered.connect(self.edaDistributions)
        edaMenu.addAction(distributionButton)

        #::--------------------------------------
        # Model Action
        # Create Random Forest option
        # Random forest text tip
        #::--------------------------------------

        randomforestButton = QAction(QIcon('Icons//forest.png'),'Random Forest', self)
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

    dfo = df.copy()

if __name__ == '__main__':
    #run voter_turnout to read in data before running GUI app
    voter_turnout()
    main()