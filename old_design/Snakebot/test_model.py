import mujoco
import mujoco.viewer
import numpy as np
import time

# Path to the MJCF file
MJCF_PATH = "scene.xml"  # Update this path if needed

# Load the model
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)

# Create a viewer for visualization
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        # print("Box z-position:", data.body_xpos[model.body_name2id("test_box"), 2])
        viewer.sync()
        time.sleep(.001)
