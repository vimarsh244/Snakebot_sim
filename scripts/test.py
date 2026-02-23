import math
import time

import mujoco
from mujoco import viewer

xml_path = "robot_desc/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("model loaded:", xml_path)
print("joints:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)])
print("actuators:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)])

# controls are kept at zero so joints do not move automatically.
# you can adjust actuators from the viewer UI sliders or use mouse perturbations.
with viewer.launch_passive(model, data) as gui:
    print("viewer launched. press ESC to quit.")
    while gui.is_running():
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        gui.sync()
        time.sleep(model.opt.timestep)
