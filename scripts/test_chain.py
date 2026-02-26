import sys
import time
import mujoco
from mujoco import viewer

num = int(sys.argv[1]) if len(sys.argv) > 1 else 3

scene_path = f"snake_description/chain_scene.xml"
chain_path = f"snake_description/chain_{num}.xml"

# generate chain xml if needed
import os
if not os.path.exists(chain_path):
    from generate_chain import generate_chain
    generate_chain(num, chain_path)

# update scene to include the right chain file
scene_xml = f"""<mujoco model="chain_scene">
    <include file="chain_{num}.xml" />
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="160" elevation="-20" />
    </visual>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
    </asset>
    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" contype="1" conaffinity="2" />
    </worldbody>
</mujoco>"""

with open(scene_path, 'w') as f:
    f.write(scene_xml)

model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

print(f"Chain loaded: {num} modules")
print(f"Joints: {model.njnt}, Actuators: {model.nu}")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {name}")

with viewer.launch_passive(model, data) as gui:
    print("viewer launched. press ESC to quit.")
    data.ctrl[:] = 0.0
    while gui.is_running():
        mujoco.mj_step(model, data)
        gui.sync()
        time.sleep(model.opt.timestep)
