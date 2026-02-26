"""Test script for the fused module"""
import time

import mujoco
from mujoco import viewer

xml_path = "robot_desc/fused_module.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("Fused module loaded successfully!")
print(f"Number of bodies: {model.nbody}")
print(f"Number of geoms: {model.ngeom}")
print(f"Number of joints: {model.njnt}")

# Launch viewer
with viewer.launch_passive(model, data) as gui:
    print("viewer launched. press ESC to quit.")
    while gui.is_running():
        mujoco.mj_step(model, data)
        gui.sync()
        time.sleep(model.opt.timestep)
