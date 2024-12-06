Install torch with <code>mayapy</code>, this specific version works with my Maya 2025:

<code>mayapy -m pip install torch==2.2.1</code>


<code>LOCAL_MAYA</code> - your local path to python modules.

**Shelf button code**

<code>
import importlib

from ML_RiggingTool import ML_RiggingTool_UI
from ML_RiggingTool import ML_RiggingTool_GenerateData
from ML_RiggingTool import ML_RiggingTool_TrainModel

importlib.reload(ML_RiggingTool_UI)
importlib.reload(ML_RiggingTool_GenerateData)
importlib.reload(ML_RiggingTool_TrainModel)


ML_RiggingTool_UI.main()
</code>
