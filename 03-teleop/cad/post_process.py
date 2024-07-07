#! /usr/bin/python3

import os
import glob
from subprocess import Popen

robot_name = "low-cost-robot"
urdf_file = f"{robot_name}/robot.urdf"

# Load the URDF file
urdf = open(urdf_file, "r")
print(f"Running from {os.getcwd()}")

# Replace all the file:///Users/raghav/Documents/projects/robot_arm/onshape-to-robot-examples/low-cost-robot-2 paths with file:// path
urdf_content = urdf.read()
urdf_content = urdf_content.replace("package://", "file:///Users/raghav/Documents/projects/robot_arm/onshape-to-robot-examples/low-cost-robot")

# change all the .stl files to .obj files
urdf_content = urdf_content.replace(".stl", ".stl.obj")

# add actuator info at the end
# first remove </robot>
urdf_content = urdf_content.replace("</robot>", "")
trasmission_template = """
    <transmission name="revolute_1_tran">
      <type>transmission_interface/SimpleTransmission</type>
      <joint name="revolute_1">
        <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      </joint>
      <actuator name="revolute_1_actr">
        <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
        <mechanicalReduction>1</mechanicalReduction>
      </actuator>
    </transmission>
"""
# Find num_joints from urdf
num_joints = urdf_content.count("<joint name=")
for i in range(1, num_joints+1):
    urdf_content += trasmission_template.replace("revolute_1", f"revolute_{i}")

urdf_content += "</robot>"


# actually convert the stl files to obj files by running
# parallel '/Users/raghav/Documents/projects/robot_arm/low_cost_robot/.venv/lib/python3.12/site-packages/stl_to_obj/__main__.py "{}" "{}.obj"' ::: *.stl

# working_dir = os.path.dirname(urdf_file)
# running_commands = []
# for stl_file in glob.glob(f"{working_dir}/*.stl"):
#     # if any of the files have (/) in their name, remove them.
#     obj_file = stl_file.replace(".stl", ".obj")
#     if not os.path.exists(obj_file):
#         print(f"Converting {stl_file} to {obj_file}")
#         running_command = Popen(f"/Users/raghav/Documents/projects/robot_arm/low_cost_robot/.venv/bin/python3 /Users/raghav/Documents/projects/robot_arm/low_cost_robot/.venv/lib/python3.12/site-packages/stl_to_obj/__main__.py {stl_file} {obj_file}", shell=True, cwd=working_dir)
#         running_commands.append(running_command)

# for running_command in running_commands:
#     running_command.wait()

# save the new file in robot-post-processed.urdf
with open(f"{robot_name}/robot-post-processed.urdf", "w") as urdf:
    urdf.write(urdf_content)

