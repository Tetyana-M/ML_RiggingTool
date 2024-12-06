import numpy as np
import maya.cmds as cmds
import os

class ML_RiggingTool_GenerateData():
    NUM_SAMPLES = 10
    TOTAL_COMBINATIONS = NUM_SAMPLES ** 6
    LOCAL_PATH = os.environ.get('LOCAL_MAYA')
    
    def __init__(self):
        pass
        
    def generateThetaCombinations(self, joint1Min, joint1Max, joint2Min, joint2Max):
        # Generate samples.
        samples = {'theta1_X': np.random.uniform(low=joint1Min[0], high=joint1Max[0], size=self.NUM_SAMPLES),
                   'theta1_Y': np.random.uniform(low=joint1Min[1], high=joint1Max[1], size=self.NUM_SAMPLES),
                   'theta1_Z': np.random.uniform(low=joint1Min[2], high=joint1Max[2], size=self.NUM_SAMPLES),
                   'theta2_X': np.random.uniform(low=joint2Min[0], high=joint2Max[0], size=self.NUM_SAMPLES),
                   'theta2_Y': np.random.uniform(low=joint2Min[1], high=joint2Max[1], size=self.NUM_SAMPLES),
                   'theta2_Z': np.random.uniform(low=joint2Min[2], high=joint2Max[2], size=self.NUM_SAMPLES)}
        # Store all theta combinations.
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

        # Generate xyz.
        end_xyz = np.empty((len(theta_combinations), 3))
        for index in range(len(theta_combinations)):
            cmds.setAttr(f'{joints[0]}.rx', theta_combinations[index][0])
            cmds.setAttr(f'{joints[0]}.ry', theta_combinations[index][1])
            cmds.setAttr(f'{joints[0]}.rz', theta_combinations[index][2])
            cmds.setAttr(f'{joints[1]}.rx', theta_combinations[index][3])
            cmds.setAttr(f'{joints[1]}.ry', theta_combinations[index][4])
            cmds.setAttr(f'{joints[1]}.rz', theta_combinations[index][5])
            pos = cmds.xform(joints[2], q=1, t=1, ws=1)
            end_xyz[index] = pos
        
        return end_xyz
        
"""
        # Specify the target directory and filename.
        directory = r'C:\Fanshawe\6147\project\data'
        theta_filename = 'theta_combinations.json'
        theta_file_path = os.path.join(directory, theta_filename)
        # Make sure the directory exists, create if it does not.
        os.makedirs(directory, exist_ok=True)




# Write theta to file.    
with open(theta_file_path, 'w') as json_file:
    json.dump(theta_combinationss, json_file, indent=4)

# Write xyz to file.    
xyz_filename = 'xyz_coordinates.json'
xyz_file_path = os.path.join(directory, xyz_filename)
with open(xyz_file_path, 'w') as json_file:
    json.dump(locator_xyz.tolist(), json_file, indent=4)

# Visualize.

# Create locator for each xyz.
for point in locator_xyz:
    #print(point)
    cmds.spaceLocator(p=(point[0], point[1], point[2]))
"""