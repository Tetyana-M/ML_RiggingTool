import numpy as np
import maya.cmds as cmds

class ML_RiggingTool_GenerateData():
    NUM_SAMPLES = 10
    
    def __init__(self):
        pass
    
    def generateThetaCombinations(self, joint1Min, joint1Max, joint2Min, joint2Max):
        """ 
        Generate combinations of X, Y, and Z rotations for joint 1 and joint2 within
        the min and max limits.
        
        @param joint1Min, joint1Max, joint2Min, joint2Max: [float, float, float] - min and max 
            X, Y, and Z rotation values for each joint respectively.
        @return theta_combinations: numpy array of (n, 6) shape where n is the number of 
            combinations and 6 is [joint1_rx, joint1_ry, joint1_rz, joint2_rx, joint2_ry, joint2_rz]
        """
        # Generate samples.
        samples = {'theta1_X': np.random.uniform(low=joint1Min[0], high=joint1Max[0], size=self.NUM_SAMPLES),
                   'theta1_Y': np.random.uniform(low=joint1Min[1], high=joint1Max[1], size=self.NUM_SAMPLES),
                   'theta1_Z': np.random.uniform(low=joint1Min[2], high=joint1Max[2], size=self.NUM_SAMPLES),
                   'theta2_X': np.random.uniform(low=joint2Min[0], high=joint2Max[0], size=self.NUM_SAMPLES),
                   'theta2_Y': np.random.uniform(low=joint2Min[1], high=joint2Max[1], size=self.NUM_SAMPLES),
                   'theta2_Z': np.random.uniform(low=joint2Min[2], high=joint2Max[2], size=self.NUM_SAMPLES)}
        # Generate theta combinations.
        theta_combinations = np.array([[samples['theta1_X'][i], 
                                         samples['theta1_Y'][j], 
                                         samples['theta1_Z'][k],
                                         samples['theta2_X'][l], 
                                         samples['theta2_Y'][m], 
                                         samples['theta2_Z'][n]]
                                         for i in range(self.NUM_SAMPLES)
                                         for j in range(self.NUM_SAMPLES)
                                         for k in range(self.NUM_SAMPLES)
                                         for l in range(self.NUM_SAMPLES)
                                         for m in range(self.NUM_SAMPLES)
                                         for n in range(self.NUM_SAMPLES)])
        
        theta_combinations = np.unique(theta_combinations, axis=0)
        
        return theta_combinations
        
    def generateEndPositions(self, joints, theta_combinations):
        """
        Get end effector positions for each combination of thetas.

        @param joints: string array of size 3 - the three joint names.
        @param theta_combinations: numpy array, shape (n,6) - X, Y, and Z rotation values for
            joint1 and joint2 formatted [joint1_rx, joint1_ry, joint1_rz, joint2_rx, joint2_ry, joint2_rz]
        @return end_xyz: numpy array of shape (n, 3) where n is the number theta combinations and 
            3 is the X, Y, and Z world space coordinates of the end effector.
        """
        # Generate xyz.
        end_xyz = np.empty((len(theta_combinations), 3))
        for index in range(len(theta_combinations)):
            cmds.setAttr(f'{joints[0]}.rx', theta_combinations[index][0])
            cmds.setAttr(f'{joints[0]}.ry', theta_combinations[index][1])
            cmds.setAttr(f'{joints[0]}.rz', theta_combinations[index][2])
            cmds.setAttr(f'{joints[1]}.rx', theta_combinations[index][3])
            cmds.setAttr(f'{joints[1]}.ry', theta_combinations[index][4])
            cmds.setAttr(f'{joints[1]}.rz', theta_combinations[index][5])
            pos = cmds.xform(joints[2], q=1, t=1, ws=1) # world space
            end_xyz[index] = pos
        
        return end_xyz
        