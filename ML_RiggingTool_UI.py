import os
from PySide6 import QtGui, QtWidgets, QtCore, QtUiTools
from shiboken6 import Shiboken

import maya.cmds as cmds
import maya.OpenMayaUI as MayaUI

from ML_RiggingTool import ML_RiggingTool_GenerateData as generateData
from ML_RiggingTool import ML_RiggingTool_TrainModel as trainModel


class ML_RiggingTool_UI(QtWidgets.QMainWindow):
    WIN_NAME = 'ML_RiggingTool'
    LOCAL_PATH = os.environ.get('LOCAL_MAYA')
    UI_PATH = os.path.realpath(f'{LOCAL_PATH}/ML_RiggingTool/ML_RiggingTool_UI.ui')
    _generatedData = generateData.ML_RiggingTool_GenerateData()
    _selectedJoints = []

    def __init__(self):
        mainUI = self.UI_PATH
        MayaMain = Shiboken.wrapInstance(int(MayaUI.MQtUtil.mainWindow()), QtWidgets.QWidget)
        super(ML_RiggingTool_UI, self).__init__(MayaMain)

        # Load UI.
        loader = QtUiTools.QUiLoader()
        uifile = QtCore.QFile(mainUI)
        uifile.open(QtCore.QFile.ReadOnly)
        ui = loader.load(uifile, MayaMain)
        uifile.close()
        self.MainWindowUI = ui

        # Get selected joints.
        self._selectedJoints = cmds.ls(sl=1,sn=True)
        if(len(self._selectedJoints) != 3):
             cmds.error('Wrong selection! Select three joints.')
        # Set joint labels in the UI.
        self.MainWindowUI.joint1_label.setText(f'Joint 1: {self._selectedJoints[0]}') 
        self.MainWindowUI.joint2_label.setText(f'Joint 2: {self._selectedJoints[1]}')
        self.MainWindowUI.joint3_label.setText(f'Joint 3: {self._selectedJoints[2]}')   

        # Validator setup.
        validator = QtGui.QDoubleValidator()
        self.MainWindowUI.joint1_rX_min_edit.setValidator(validator)
        self.MainWindowUI.joint1_rY_min_edit.setValidator(validator)
        self.MainWindowUI.joint1_rZ_min_edit.setValidator(validator)
        self.MainWindowUI.joint1_rX_max_edit.setValidator(validator)
        self.MainWindowUI.joint1_rY_max_edit.setValidator(validator)
        self.MainWindowUI.joint1_rZ_max_edit.setValidator(validator)
        self.MainWindowUI.joint2_rX_min_edit.setValidator(validator)
        self.MainWindowUI.joint2_rY_min_edit.setValidator(validator)
        self.MainWindowUI.joint2_rZ_min_edit.setValidator(validator)
        self.MainWindowUI.joint2_rX_max_edit.setValidator(validator)
        self.MainWindowUI.joint2_rY_max_edit.setValidator(validator)
        self.MainWindowUI.joint2_rZ_max_edit.setValidator(validator)
        
        # Button setup.
        self.MainWindowUI.createControl_button.clicked.connect(self.createControl)

        # Show UI.
        self.MainWindowUI.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.MainWindowUI.show()        
        
    def createControl(self):
        # Get min and max values.
        joint1Min = [float(self.MainWindowUI.joint1_rX_min_edit.text()),
                     float(self.MainWindowUI.joint1_rY_min_edit.text()),
                     float(self.MainWindowUI.joint1_rZ_min_edit.text())]
        joint1Max = [float(self.MainWindowUI.joint1_rX_max_edit.text()),
                     float(self.MainWindowUI.joint1_rY_max_edit.text()),
                     float(self.MainWindowUI.joint1_rZ_max_edit.text())]
        joint2Min = [float(self.MainWindowUI.joint2_rX_min_edit.text()),
                     float(self.MainWindowUI.joint2_rY_min_edit.text()),
                     float(self.MainWindowUI.joint2_rZ_min_edit.text())]
        joint2Max = [float(self.MainWindowUI.joint2_rX_max_edit.text()),
                     float(self.MainWindowUI.joint2_rY_max_edit.text()),
                     float(self.MainWindowUI.joint2_rZ_max_edit.text())]
        #TEMP
        joint1Min = [-100.0, 0.0, -20.0]
        joint1Max = [60.0, 0.0, 95.0]
        joint2Min = [0.0, 0.0, 0.0]
        joint2Min = [110.0, 0.0, 0.0]
         
        theta_combinations = self._generatedData.generateThetaCombinations(joint1Min, joint1Max, joint2Min, joint2Max)
        end_xyz = self._generatedData.generateEndPositions(self._selectedJoints, theta_combinations)

        trainModel.ML_RiggingTool_TrainModel(theta_combinations, end_xyz)
        
def main():
        winName = ML_RiggingTool_UI.WIN_NAME
        if cmds.window(winName, query=True, exists=True):
             cmds.deleteUI(winName)

        ML_RiggingTool_UI()
        
