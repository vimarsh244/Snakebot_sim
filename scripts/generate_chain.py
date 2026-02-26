"""
Generate MJCF XML for a chain of N snakebot modules with battery holders in between.

Usage:
    python scripts/generate_chain.py <num_modules> [output_path]
    
Example:
    python scripts/generate_chain.py 3 bestRenamed/chain_3.xml
    
Pattern for 3 modules:
    module_1 - battery_1 - module_2 - battery_2 - module_3 - battery_3
"""

import sys
import math
import textwrap

# chain pitch: distance between consecutive modules along Z (original vertical frame)
# module extent (bottom surface to top surface) + battery thickness
PITCH = 0.1359

# battery body position relative to top plate body (as child)
# rotated 180° around the snake axis (Z) so the bracket faces correctly
BATTERY_POS_IN_TOP_PLATE = "0.07386 0.049385 -0.03101"
BATTERY_EULER_IN_TOP_PLATE = "0 0 4.71238898038469"

# battery approximate inertial properties
BATTERY_MASS = 0.15
BATTERY_INERTIAL_POS = "0.033 -0.083 0.059"
BATTERY_DIAGINERTIA = "0.000172 0.000153 0.000306"

# frame settings for horizontal orientation
FRAME_Z = 0.08
FRAME_EULER = f"0 {math.pi/2} 0"


def module_xml(n, z_offset):
    """Generate XML for module n (1-indexed) with all bodies shifted by z_offset along Z."""
    p = f"m{n}_"

    bp_z = 0.0145310940675635 + z_offset
    tp_z = 0.09490695867371797 + z_offset

    return textwrap.dedent(f"""\
            <body name="{p}bottom-base-plate-v1" pos="0.0 0.033146846605945184 {bp_z}" euler="-3.141592653589787 -0.0 0.0">
                <freejoint name="{p}bottom_plate_free" />
                <geom name="{p}bottom-base-plate-v1_geom" type="mesh" mesh="bottom-base-plate-v1" pos="0 0 0" euler="0 0 0" />
                <inertial mass="0.6346416371043081" pos="-0.005347637848628755 0.017762619548364243 0.005846593790389482" fullinertia="0.00035240661436305326 0.0004045289246972485 0.0006907997214385136 2.0631258195209793e-05 1.3728562784376835e-05 2.136024439408237e-06" />
                <body name="{p}bottom-diagonal-chintu-v1" pos="0.04926263546510042 -0.04140998343068075 0.009999457199693682" euler="-1.5603770637091834 3.3923831869601494e-14 -0.7853981633974351">
                    <joint name="{p}Revolute-8" type="hinge" axis="-1.0000003094489522 3.8658312662143146e-14 1.2434497875801753e-14" pos="-0.0425002751358633 0.009999904807428087 -1.0747999264540658e-07" />
                    <geom name="{p}bottom-diagonal-chintu-v1_geom" type="mesh" mesh="bottom-diagonal-chintu-v1" pos="0 0 0" euler="0 0 0" />
                    <inertial mass="0.02683295322437194" pos="-0.03749999999999955 0.011871044300627112 -7.670762931010015e-05" fullinertia="2.583516087012673e-06 1.6570207418707578e-06 1.3643364476595977e-06 -7.783679114529989e-19 -2.3414566936227997e-19 5.3236682858778105e-09" />
                </body>
                <body name="{p}bottom-side-chintu-v1" pos="-0.00799999999999995 0.016966177185297776 0.009999942800696776" euler="-1.5674140388758464 9.144902985720049e-17 -8.872539658711765e-15">
                    <joint name="{p}Revolute-9" type="hinge" axis="1.0 -6.143908029357409e-17 8.872798214762224e-15" pos="-0.03250000000000005 0.009999905414152896 -1.5307501481837577e-07" />
                    <geom name="{p}bottom-side-chintu-v1_geom" type="mesh" mesh="bottom-side-chintu-v1" pos="0 0 0" euler="0 0 0" />
                    <inertial mass="0.02683295322437194" pos="-0.03749999999999956 0.011871044300627118 -7.670762931009596e-05" fullinertia="2.5835160870126677e-06 1.6570207418707627e-06 1.3643364476595892e-06 -7.869942493139945e-19 -2.3724382550196554e-19 5.323668285882103e-09" />
                    <body name="{p}danda-top-servo-v1" pos="-0.001039225735187278 0.030188428968929436 -0.0001000000000005552" euler="-3.141592653589793 -3.094550169498861e-15 -3.136427452099049">
                        <joint name="{p}Revolute-13" type="hinge" axis="1.4489308988763791e-09 -2.805176106089602e-07 0.9999997189799212" pos="0.036499999974189734 -0.007499995003343646 -0.007499610395922852" />
                        <geom name="{p}danda-top-servo-v1_geom" type="mesh" mesh="danda-top-servo-v1" pos="0 0 0" euler="0 0 0" />
                        <inertial mass="0.0922506301325943" pos="-2.3784152288882894e-05 0.02001791956679029 -0.01786461691933594" fullinertia="4.2350317380439685e-05 9.535228719088794e-05 0.00010032395639346058 4.066768036356792e-05 3.919681021637052e-08 -2.9531843309192267e-08" />
                    </body>
                </body>
                <body name="{p}bottom-chintu-and-servo-head-v1" pos="-0.00909307013060929 0.01600000000000013 0.009999693148195961" euler="-1.5629623830476533 -1.0643858845815612e-16 -1.5707963267949052">
                    <joint name="{p}Revolute-15" type="hinge" axis="-1.0 6.33541605345897e-15 -8.555451894091773e-15" pos="-0.031599846605945045 0.009999903831887062 -2.6776042357766494e-07" />
                    <geom name="{p}bottom-chintu-and-servo-head-v1_geom" type="mesh" mesh="bottom-chintu-and-servo-head-v1" pos="0 0 0" euler="0 0 0" />
                    <inertial mass="0.03185386237967729" pos="-0.03650460388444623 0.011576204158460035 -6.457349709795574e-05" fullinertia="2.8347809965994143e-06 1.948930129997281e-06 1.6711645787283958e-06 4.99611700119328e-08 -2.0535459295348652e-09 5.93145030151227e-09" />
                    <body name="{p}proxy_Rev6" pos="-0.0374998466 0.0225002604 0.0074000332">
                        <joint name="{p}Revolute-6" type="hinge" axis="0 0 -1" />
                        <inertial mass="1e-6" pos="0 0 0" diaginertia="1e-12 1e-12 1e-12" />
                    </body>
                </body>
            </body>
            <body name="{p}top-base-plate-v1" pos="-0.025385058611262133 0.006860259759015023 {tp_z}" euler="-4.031953765759422e-12 -2.2077461834844432e-13 -1.5707963267948715">
                <freejoint name="{p}top_plate_free" />
                <geom name="{p}top-base-plate-v1_geom" type="mesh" mesh="top-base-plate-v1" pos="0 0 0" euler="0 0 0" />
                <inertial mass="0.6346416371043081" pos="-0.005347637848628756 0.01776261954836422 0.005846593790389466" fullinertia="0.0003524066143630541 0.0004045289246972487 0.000690799721438512 2.0631258195209773e-05 1.3728562784376767e-05 2.1360244394083527e-06" />
                <body name="{p}top-side-chintu-v1" pos="-0.00799999999999995 0.017078338636143844 0.00999969314819694" euler="-1.578630270538108 -1.306137021765825e-16 -8.880599230995793e-15">
                    <joint name="{p}Revolute-4" type="hinge" axis="1.0 2.2102639645820695e-13 -1.4600991065996208e-14" pos="-0.042499740240984595 0.009999958215754213 5.828571949048492e-08" />
                    <geom name="{p}top-side-chintu-v1_geom" type="mesh" mesh="top-side-chintu-v1" pos="0 0 0" euler="0 0 0" />
                    <inertial mass="0.02683295322437194" pos="-0.03749999999999955 0.011871044300627109 -7.670762931009832e-05" fullinertia="2.583516087012683e-06 1.6570207418708078e-06 1.3643364476596796e-06 -7.692487408204518e-19 -2.366797700599352e-19 5.323668285878194e-09" />
                    <body name="{p}danda-bottom-servo-v1-v1" pos="-0.0010394379249127627 0.030189435165771963 -0.00010000000000055742" euler="3.141592653589793 -2.9614914578805773e-15 -3.136399855324415">
                        <joint name="{p}Revolute-5" type="hinge" axis="7.20995457049534e-10 -1.388417767162614e-07 0.9999996862584414" pos="0.03649973832316588 -0.007499631354740782 -0.007500211881852813" />
                        <geom name="{p}danda-bottom-servo-v1-v1_geom" type="mesh" mesh="danda-bottom-servo-v1-v1" pos="0 0 0" euler="0 0 0" />
                        <inertial mass="0.0922506301325943" pos="-2.3784152288883613e-05 0.02001791956679029 -0.017864616919335944" fullinertia="4.235031738043959e-05 9.535228719088794e-05 0.00010032395639346057 4.06676803635679e-05 3.919681021637007e-08 -2.953184330917612e-08" />
                    </body>
                </body>
                <body name="{p}top-diagonal-chintu-v1" pos="-0.0036230251583287965 0.011770373123872933 0.009999457199694799" euler="-1.5603770637118737 6.644630040307432e-15 2.356194490192345">
                    <joint name="{p}Revolute-7" type="hinge" axis="1.0000003094489522 -3.013877342139537e-12 6.2727600891321345e-15" pos="-0.03249975863110075 0.009999956330450638 -2.246789806976512e-07" />
                    <geom name="{p}top-diagonal-chintu-v1_geom" type="mesh" mesh="top-diagonal-chintu-v1" pos="0 0 0" euler="0 0 0" />
                    <inertial mass="0.02683295322437194" pos="-0.03749999999999956 0.011871044300627121 -7.67076293101015e-05" fullinertia="2.5835160870126495e-06 1.6570207418707642e-06 1.3643364476595733e-06 -7.923024196166957e-19 -2.5458600821209385e-19 5.323668285889434e-09" />
                    <body name="{p}danda-support-v1" pos="0.007299769040579557 0.07018889486988192 -0.00010000000000057963" euler="3.141592653589793 3.0759154446592663e-15 0.004207017307401544">
                        <joint name="{p}Revolute-11" type="hinge" axis="-2.8891292943510017e-10 6.86758660859571e-08 0.999999434404338" pos="-0.044999711787780526 0.04750026372664425 -0.007499920962069091" />
                        <geom name="{p}danda-support-v1_geom" type="mesh" mesh="danda-support-v1" pos="0 0 0" euler="0 0 0" />
                        <inertial mass="0.10848592066715428" pos="-2.9218977497941223e-15 0.019999999999999987 -0.023051676135472255" fullinertia="5.76472579093131e-05 0.00016423658507161952 0.0001583087957754129 5.5901833496867315e-05 7.27961290694027e-18 -6.183340514956392e-20" />
                        <body name="{p}proxy_Rev10" pos="0.0449996782 -0.0075003154 0.0075000382">
                            <joint name="{p}Revolute-10" type="hinge" axis="0 0 1" />
                            <inertial mass="1e-6" pos="0 0 0" diaginertia="1e-12 1e-12 1e-12" />
                        </body>
                    </body>
                </body>
                <body name="{p}top-chintu-and-servo-head-v1" pos="-0.00898090867971997 0.015099999999999631 0.009999942800695454" euler="-1.5741786147141739 4.058338165747889e-14 -1.5707963267948932">
                    <joint name="{p}Revolute-16" type="hinge" axis="-1.0 -3.991444155681031e-12 1.5061075503405146e-14" pos="-0.03250005861126232 0.009999958703343846 -8.886286865848563e-09" />
                    <geom name="{p}top-chintu-and-servo-head-v1_geom" type="mesh" mesh="top-chintu-and-servo-head-v1" pos="0 0 0" euler="0 0 0" />
                    <inertial mass="0.03185386237967729" pos="-0.03650460388444623 0.011576204158460028 -6.457349709795596e-05" fullinertia="2.834780996599465e-06 1.9489301299972826e-06 1.6711645787284378e-06 4.99611700119453e-08 -2.05354592953819e-09 5.931450301505956e-09" />
                    <body name="{p}proxy_Rev14" pos="-0.0375000586 0.0224997347 -0.0075997746">
                        <joint name="{p}Revolute-14" type="hinge" axis="0 0 -1" />
                        <inertial mass="1e-6" pos="0 0 0" diaginertia="1e-12 1e-12 1e-12" />
                    </body>
                </body>
                <body name="battery_{n}" pos="{BATTERY_POS_IN_TOP_PLATE}" euler="{BATTERY_EULER_IN_TOP_PLATE}">
                    <geom name="battery_{n}_geom" type="mesh" mesh="battery_middle_module_v2" rgba="0.3 0.3 0.35 1" />
                    <inertial mass="{BATTERY_MASS}" pos="{BATTERY_INERTIAL_POS}" diaginertia="{BATTERY_DIAGINERTIA}" />
                </body>
            </body>""")


def equality_xml(n):
    """Generate weld equality constraints for module n."""
    p = f"m{n}_"
    return textwrap.dedent(f"""\
        <weld name="{p}loop_Rev6" body1="{p}proxy_Rev6" body2="{p}danda-bottom-servo-v1-v1" solref="0.0005 1" solimp="0.9999 0.99999 0.00001" />
        <weld name="{p}loop_Rev10" body1="{p}proxy_Rev10" body2="{p}bottom-diagonal-chintu-v1" solref="0.0005 1" solimp="0.9999 0.99999 0.00001" />
        <weld name="{p}loop_Rev14" body1="{p}proxy_Rev14" body2="{p}danda-top-servo-v1" solref="0.0005 1" solimp="0.9999 0.99999 0.00001" />""")


def actuator_xml(n):
    """Generate actuators for module n."""
    p = f"m{n}_"
    return textwrap.dedent(f"""\
        <position name="module_{n}_servo_bottom" joint="{p}Revolute-15" kp="25" ctrlrange="-12.56637 12.56637" />
        <position name="module_{n}_servo_top" joint="{p}Revolute-16" kp="25" ctrlrange="-12.56637 12.56637" />""")


def generate_chain(num_modules, output_path):
    lines = []
    lines.append(f'<mujoco model="snakebot_chain_{num_modules}">')
    lines.append('    <compiler angle="radian" eulerseq="XYZ" />')
    lines.append('')
    lines.append('    <option timestep="0.0005" iterations="500" solver="Newton" impratio="10" tolerance="1e-14" noslip_iterations="50" />')
    lines.append('')
    lines.append('    <default>')
    lines.append('        <joint damping="0.5" armature="0.01" />')
    lines.append('        <geom contype="2" conaffinity="1" />')
    lines.append('    </default>')
    lines.append('')

    # assets (shared across all modules)
    lines.append('    <asset>')
    mesh_names = [
        "bottom-base-plate-v1", "top-base-plate-v1", "danda-support-v1",
        "danda-top-servo-v1", "danda-bottom-servo-v1-v1",
        "bottom-chintu-and-servo-head-v1", "top-chintu-and-servo-head-v1",
        "top-diagonal-chintu-v1", "bottom-side-chintu-v1",
        "top-side-chintu-v1", "bottom-diagonal-chintu-v1",
    ]
    for name in mesh_names:
        lines.append(f'        <mesh name="{name}" file="meshes/{name}.stl" scale="0.001 0.001 0.001" />')
    lines.append(f'        <mesh name="battery_middle_module_v2" file="meshes/battery_middle_module_v2.stl" scale="0.001 0.001 0.001" />')
    lines.append('    </asset>')

    # worldbody with frame for horizontal orientation
    lines.append('    <worldbody>')
    lines.append(f'        <frame pos="0 0 {FRAME_Z}" euler="{FRAME_EULER}">')

    for n in range(1, num_modules + 1):
        z_offset = (n - 1) * PITCH
        lines.append(f'            <!-- module {n} -->')
        lines.append(module_xml(n, z_offset))

    lines.append('        </frame>')
    lines.append('    </worldbody>')

    # equality constraints
    lines.append('    <equality>')
    for n in range(1, num_modules + 1):
        lines.append(equality_xml(n))
    # inter-module welds: battery_n -> next module's bottom plate
    for n in range(1, num_modules):
        lines.append(f'        <weld name="chain_link_{n}" body1="battery_{n}" body2="m{n+1}_bottom-base-plate-v1" solref="0.0005 1" solimp="0.9999 0.99999 0.00001" />')
    lines.append('    </equality>')

    # actuators
    lines.append('    <actuator>')
    for n in range(1, num_modules + 1):
        lines.append(actuator_xml(n))
    lines.append('    </actuator>')

    lines.append('</mujoco>')

    xml = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(xml)

    print(f"Generated {output_path} with {num_modules} modules")
    print(f"  Actuators: {num_modules * 2} (module_N_servo_top, module_N_servo_bottom)")
    print(f"  Chain length: ~{num_modules * PITCH * 1000:.0f}mm horizontal")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <num_modules> [output_path]")
        sys.exit(1)

    num = int(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else f"bestRenamed/chain_{num}.xml"
    generate_chain(num, out)
