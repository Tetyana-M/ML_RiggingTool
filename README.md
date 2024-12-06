Install torch with <code>mayapy</code>, this specific version works with my Maya 2025:

<code>mayapy -m pip install torch==2.2.1</code>


<code>LOCAL_MAYA</code> - your local path to python modules.

**Shelf button code:**
<code>
import importlib<br>
from ML_RiggingTool import ML_RiggingTool_UI<br>
from ML_RiggingTool import ML_RiggingTool_GenerateData<br>
from ML_RiggingTool import ML_RiggingTool_TrainModel<br><br>
importlib.reload(ML_RiggingTool_UI)<br>
importlib.reload(ML_RiggingTool_GenerateData)<br>
importlib.reload(ML_RiggingTool_TrainModel)<br><br>
ML_RiggingTool_UI.main()
</code>
