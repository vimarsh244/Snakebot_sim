import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
robotIDs = p.loadSDF("best.sdf", useMaximalCoordinates=1)
print("Loaded robot body IDs:", robotIDs)
for i in range(240):
    p.stepSimulation()
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotIDs[0])
print("Position:", cubePos)
print("Orientation:", cubeOrn)
p.disconnect()
print("Done.")
