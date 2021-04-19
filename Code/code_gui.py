import pandas as pd
from pathlib import Path
import os
import sys


from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, \
                            QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Figure
import matplotlib.pyplot as plt

#get directory path
dir = str(Path(os.getcwd()).parents[0])



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

class BarChartPlot(QMainWindow):
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

        super(BarChartPlot, self).__init__()

        self.Title = "Distribution of Responses by Demographic"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)

        self.fig = Figure()


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
        modelMenu = mainMenu.addMenu('Random Forest')

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
    # read in data
    global demographics
    df = pd.read_csv(dir + '\\' + 'nonvoters_data.csv', sep=',', header=0)
    demographics = df['educ', 'race', 'gender', 'income']

if __name__ == '__main__':
    main()



