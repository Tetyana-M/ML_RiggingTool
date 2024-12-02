import maya.cmds as cmds
import maya.OpenMayaUI as moui
from functools import partial
import PySide6.QtCore as QtCore
import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtWidgets
import sys
import consts as consts
from shiboken2 import wrapInstance
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin

def getMainWindow():
    mainWindowPtr = moui.MQtUtil.mainWindow()
    mainWindow = wrapInstance(long(mainWindowPtr), QtWidgets.QWidget)
    return mainWindow


class testWindow(MayaQWidgetDockableMixin, QtWidgets.QDialog):
    """ Test tool class.
    """

    def __init__(self):
        super(testWindow, self).__init__(getMainWindow())

        self.setWindowTitle(consts.WINDOW_TITLE)
        self.resize(consts.WINDOW_W, consts.WINDOW_H)
        self.table_widget = MyTableWidget(parent_instance=self)
        self.table_widget.resize(consts.WINDOW_W,consts.WINDOW_H)

        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)

class MyTableWidget(QtWidgets.QWidget):
    """ This table puts together Picker and Walker into tabs.
    """
    leftLegButtonStylesheet = "QPushButton{border-style:outset;border-width:2px;border-radius:15px;border-color:QColor(106,134,158);font:14px;padding:3px;}"
    leftLegButtonStylesheet = leftLegButtonStylesheet + "QPushButton::hover{color:white;border-style:outset;font:bold 14px}"
    rightLegButtonStylesheet = "QPushButton{border-style:outset;border-width:2px;border-radius:15px;border-color:QColor(200,100,170);font:14px;padding:3px;}"
    rightLegButtonStylesheet = rightLegButtonStylesheet + "QPushButton::hover{color:white;border-style:outset;font:bold 14px}"
    bodyButtonStylesheet = "QPushButton{border-style:outset;border-width:2px;border-radius:30px;border-color:QColor(252, 236, 227);font:14px;padding:3px;}"
    bodyButtonStylesheet = bodyButtonStylesheet + "QPushButton::hover{color:white;border-style:outset;font:bold 14px}"
    abdomenButtonStylesheet = "QPushButton{border-style:outset;border-width:2px;border-radius:12px;border-color:QColor(252,236,227);font:14px;padding:3px;}"
    abdomenButtonStylesheet = abdomenButtonStylesheet + "QPushButton::hover{color:white;border-style:outset;font:bold 14px}"
    setKeyframesButtonStylesheet = "QPushButton{border-style:outset;border-width:2px;border-radius:12px;border-color:QColor(252,236,227);font:10px;padding:3px;}"
    setKeyframesButtonStylesheet = setKeyframesButtonStylesheet + "QPushButton::hover{color:white;border-style:outset;font:bold 11px}"
    toolButtonStylesheet = "QPushButton{border-style:none;border-width:2px;}"
    toolButtonStylesheet = toolButtonStylesheet + "QPushButton::hover{border-style:outset;border-width:2px;}"
    labelLeftStylesheet = "font:bold 14px;color:QColor(106,134,158);"
    labelRightStylesheet = "font:bold 14px;color:QColor(200,100,170);"
    leftSelectionButtonStylesheet = "QPushButton{font:10px;border-style:outset;border-width:2px;border-color:QColor(106,134,158);}"
    leftSelectionButtonStylesheet = leftSelectionButtonStylesheet + "QPushButton::hover{border-style:outset;font:bold 10px}"
    rightSelectionButtonStylesheet = "QPushButton{font:10px;border-style:outset;border-width:2px;border-color:QColor(200,100,170);}"
    rightSelectionButtonStylesheet = rightSelectionButtonStylesheet + "QPushButton::hover{border-style:outset;font:bold 10px}"
    centerSelectionButtonStylesheet = "QPushButton{font:10px;border-style:outset;border-width:2px;border-color:QColor(252, 236, 227);}"
    centerSelectionButtonStylesheet = centerSelectionButtonStylesheet + "QPushButton::hover{border-style:outset;font:bold 10px}"

    picker_help_file = "C:/Users/Tania/Documents/maya/modules/spider/picker_help_text.html"
    walker_help_file = "C:/Users/Tania/Documents/maya/modules/spider/walker_help_text.html"

    def __init__(self, parent_instance):
        super(MyTableWidget, self).__init__(parent_instance)
        self.tabs_layout = QtWidgets.QVBoxLayout(self)

        # Initialize tabs
        self.tabs = QtWidgets.QTabWidget()

        self.picker_tab = QtWidgets.QWidget()
        self.picker_tab.setStyleSheet("background-image: url({tname});".format(tname=consts.BACKGROUND))

        self.walker_tab = QtWidgets.QWidget()
        self.walker_tab.setStyleSheet("background-image: url({tname});".format(tname=consts.BACKGROUND))

        self.help_tab = QtWidgets.QWidget()

        # Add tabs
        self.tabs.addTab(self.picker_tab, "Picker")
        self.tabs.addTab(self.walker_tab, "Walker")
        self.tabs.addTab(self.help_tab, "Help")

        # Create Picker tab
        self.picker_tab.boxLayout = QtWidgets.QVBoxLayout(self)
        self.picker_widget = MyPickerWidget(parent_widget=self)
        self.picker_tab.boxLayout.addWidget(self.picker_widget)
        self.picker_tab.setLayout(self.picker_tab.boxLayout)
        self.picker_tab.boxLayout.setAlignment(QtCore.Qt.AlignTop)

        # Create Walker tab
        self.walker_tab.boxLayout = QtWidgets.QVBoxLayout(self)
        self.walker_widget = MyWalkerWidget(parent_widget=self)
        self.walker_tab.boxLayout.addWidget(self.walker_widget)
        self.walker_tab.setLayout(self.walker_tab.boxLayout)
        self.walker_tab.boxLayout.setAlignment(QtCore.Qt.AlignTop)

        # Create Help tab
        self.help_tab.boxLayout = QtWidgets.QVBoxLayout(self)
        self.help_widget = MySpiderHelpWidget(parent_widget=self)
        self.help_tab.boxLayout.addWidget(self.help_widget)
        self.help_tab.setLayout(self.help_tab.boxLayout)
        self.help_tab.boxLayout.setAlignment(QtCore.Qt.AlignTop)

        # Create Close button
        self.CloseButton = QtWidgets.QPushButton("Close")
        self.CloseButton.clicked.connect(parent_instance.close)


        # Add tabs to widget
        self.tabs_layout.addWidget(self.tabs)
        self.tabs_layout.addWidget(self.CloseButton)
        self.setLayout(self.tabs_layout)

def start():
    global spider_ui
    try:
        spider_ui.close()
        spider_ui.deleteLater()
    except:
        pass
    spider_ui = spiderPickerWalker()
    spider_ui.show(dockable=True, floating=True)

if __name__ == "__main__":
    start()

