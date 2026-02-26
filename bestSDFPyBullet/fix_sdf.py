import xml.etree.ElementTree as ET
import shutil, os

path = '/mnt/c/Users/Vimarsh/Desktop/ERC/Snakebot/bestSDFPyBullet/best.sdf'

# Backup original
if not os.path.exists(path + '.bak'):
    shutil.copy(path, path + '.bak')

tree = ET.parse(path)
root = tree.getroot()
model = root.find('model')

# Joints that create closed loops (each of these makes a link have 2 parents)
# We remove one joint from each loop to break it and form a valid tree.
joints_to_remove = {
    'top-side-chintu-v1_Revolute-5',       # danda-bottom-servo-v1-v1 gets 2nd parent removed
    'bottom-side-chintu-v1_Revolute-13',   # danda-top-servo-v1 gets 2nd parent removed
    'danda-support-v1_Revolute-10',        # bottom-diagonal-chintu-v1 gets 2nd parent removed
}

for jname in list(joints_to_remove):
    for j in model.findall('joint'):
        if j.get('name') == jname:
            model.remove(j)
            print(f'Removed loop-closing joint: {jname}')
            break

# The two roots (bottom-base-plate-v1 and top-base-plate-v1) are still disconnected.
# Connect them with a fixed joint so there is one single root.
fixed_joint = ET.SubElement(model, 'joint')
fixed_joint.set('name', 'base_connect_fixed')
fixed_joint.set('type', 'fixed')
pose_el = ET.SubElement(fixed_joint, 'pose')
pose_el.text = '0 0 0 0 0 0'
parent_el = ET.SubElement(fixed_joint, 'parent')
parent_el.text = 'bottom-base-plate-v1'
child_el = ET.SubElement(fixed_joint, 'child')
child_el.text = 'top-base-plate-v1'
print('Added fixed joint: bottom-base-plate-v1 -> top-base-plate-v1')

print('\nFinal joint tree:')
children = {}
for j in model.findall('joint'):
    p = j.find('parent').text
    c = j.find('child').text
    children.setdefault(p, []).append((j.get('name'), c))

def print_tree(node, indent=0):
    for jname, child in children.get(node, []):
        print(f'{"  "*indent}{node} --[{jname}]--> {child}')
        print_tree(child, indent+1)

print_tree('bottom-base-plate-v1')

tree.write(path, xml_declaration=True, encoding='utf-8')
print('\nSDF saved successfully.')
