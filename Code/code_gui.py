import pandas as pd
from pathlib import Path
import os
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication
from PyQt5.QtGui import QIcon

#get directory path
dir = str(Path(os.getcwd()).parents[0])

#read in data
df = pd.read_csv(dir+'\\'+'nonvoters_data.csv', sep=',', header=0)
print(df.head)

#gui
class Menu(QMainWindow):

    def __init__(self):
        super().__init__()
        #::-----------------------
        #:: size of gui window
        #::-----------------------
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 300

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
        self.statusBar()
        #::-----------------------------
        # Menu bar items
        #::-----------------------------
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        edaMenu = mainMenu.addMenu('EDA')
        modelMenu = mainMenu.addMenu('Random Forest')

        #::--------------------------------------
        # Exit action
        # The following code creates the the Exit Action along
        # with all the characteristics associated with the action
        # The Icon, a shortcut , the status tip that would appear in the window
        # and the action
        #  triggered.connect will indicate what is to be done when the item in
        # the menu is selected
        # These definitions are not available until the button is assigned
        # to the menu
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), '&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        #:: This line adds the button (item element ) to the menu

        fileMenu.addAction(exitButton)

        #:: This line shows the windows

        self.show()


#::------------------------
#:: Application starts here
#::------------------------
def main():
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = Menu()  # Creates the menu
    sys.exit(app.exec_())  # Close the application


if __name__ == '__main__':
    main()



