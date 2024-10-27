from time import sleep
import numpy as np
from matplotlib import pyplot as plt
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BasicVector,
    ConstantValueSource,
    ConstantValueSource,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    MeshcatVisualizer,
    Parser,
    Rgba,
    Simulator,
    InverseDynamicsController,
    LeafSystem,
    JointSliders,
    Meshcat,
    FirstOrderLowPassFilter,
    TrajectorySource,
    VectorLogSink
)
import numpy as np

from robot import Robot
from dynamixel import Dynamixel
import click
from utils import TruncateVec, ZeroExtendVec
import logging
from pinv import PseudoInverseController
from utils import RenderDiagram
from gripper_frames import make_gripper_frames, make_gripper_trajectory

logger = logging.getLogger("robot")
logging.basicConfig(level=logging.INFO)

@click.group()
def cli():
    pass

class PrintJacobian(LeafSystem):
    def __init__(self, plant, frame):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._frame = frame
        self.DeclareVectorInputPort("state", plant.num_multibody_states())
        self.DeclareForcedPublishEvent(self.Publish)

    def Publish(self, context):
        state = self.get_input_port().Eval(context)
        self._plant.SetPositionsAndVelocities(self._plant_context, state)
        W = self._plant.world_frame()
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._frame,
            [0, 0, 0],
            W,
            W,
        )  ## This is the important line

        print("J_G:")
        print(np.array2string(J_G, formatter={"float": lambda x: "{:5.1f}".format(x)}))
        print(
            f"smallest singular value(J_G): {np.min(np.linalg.svd(J_G, compute_uv=False))}"
        )


def build_sim_robot(meshcat):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-2)
    # Note that we parse into both the plant and the scene_graph here.
    robot_model = Parser(plant, scene_graph).AddModelsFromUrl(
            "file:///Users/raghav/Documents/projects/robot_arm/onshape-to-robot-examples/low-cost-robot/robot-post-processed.urdf"
    )[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("link_0"))
    plant.Finalize()
    
    low_cost_robot = plant.GetModelInstanceByName("onshape")
        
    kp = [100] * plant.num_positions()
    ki = [1] * plant.num_positions()
    kd = [10] * plant.num_positions()
    static_controller = builder.AddSystem(InverseDynamicsController(plant, kp, ki, kd, False))
    extend_pos = builder.AddSystem(ZeroExtendVec(6, 12))
    truncate_state = builder.AddSystem(TruncateVec(12, 6))

    builder.ExportInput(extend_pos.get_input_port(), "desired_position")
    builder.ExportOutput(truncate_state.get_output_port(), "measured_position")

    # Connect the robot state to the ID controller
    builder.Connect(
        plant.get_state_output_port(low_cost_robot),
        static_controller.get_input_port_estimated_state(),
    )

    # Connect the ID output to the plant
    builder.Connect(
        static_controller.get_output_port_control(), 
        plant.get_actuation_input_port()
    )

    builder.Connect(
        extend_pos.get_output_port(),
        static_controller.get_input_port(1)
    )

    builder.Connect(
        plant.get_state_output_port(low_cost_robot),
        truncate_state.get_input_port()
    )

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    diagram.set_name("low-cost-robot (sim)")
    return diagram, plant

class ComputeEePosition(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("joint_position", 6)
        self.DeclareVectorOutputPort("ee_position", 3, self.CalcOutput)
        self._plant_context = plant.CreateDefaultContext()
        self._plant = plant

    def CalcOutput(self, context, outputs):
        joint_positions = self.GetInputPort("joint_position").Eval(context)
        self._plant.SetPositions(self._plant_context, joint_positions)
        plant_pose = self._plant.get_body_poses_output_port().Eval(self._plant_context)
        gripper_pose = plant_pose[7]
        outputs.SetFromVector(gripper_pose.translation())



class BasicVelocityPlanner(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("desired_ee_position", 3)
        self.DeclareVectorInputPort("current_ee_position", 3)
        self.DeclareVectorOutputPort("desired_ee_velocity", 6, self.CalcOutput)

    def CalcOutput(self, context, outputs):
        desired_position: np.ndarray = self.GetInputPort("desired_ee_position").Eval(context) # type: ignore
        current_position: np.ndarray = self.GetInputPort("current_ee_position").Eval(context) # type: ignore
        position_velocity = desired_position - current_position
        angular_velocity = [0, 0, 0]
        velocity = np.concatenate([position_velocity, angular_velocity])

        logger.info(f"Desired pos: {desired_position}, current pos: {current_position}, velocity: {velocity}")
        # compute the desired velocity
        outputs.SetFromVector(velocity) # type: ignore

class LowCostRobot(LeafSystem):
    def __init__(self, plant, num_movable_joints=6):
        LeafSystem.__init__(self)
        dynamixel = Dynamixel.Config(
            baudrate=57_600,
            device_name='/dev/tty.usbmodem578E0211421'
        ).instantiate()
        self._robot = Robot(dynamixel, servo_ids=[9,1,2,3,4,5])
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._plant_robot = plant.GetModelInstanceByName("onshape")
        # self._robot._disable_torque()
        pos = self._robot.read_position()
        if not pos:
            raise Exception("Could not connect to bot")

        self._num_movable_joints = num_movable_joints

        # Slowly move the robot to the home position
        self._home_position = [2048, 1024, 2048, 2048, 2048, 2367]
        self._robot.set_goal_pos(self._home_position)

        self.DeclareVectorInputPort("desired_position", 6) # type: ignore
        self.DeclareVectorOutputPort("measured_position", 6, self.CalcOutput) # type: ignore
        

    def CalcOutput(self, context, outputs):
        # should i be setting the plant pos here? hmmm -- no because teh pinv controller does that to calculate the jacobian. (future raghav) eep yes, don't do it in the jacobian why!!
        # first sanitize the input port. just constrain everything from pi/ 3 to -pi / 3.
        # eventually constrain based on fk result
        # next convert [-pi, pi] to [0, 4096]
        # desired position shouldn't be too far from current position (no difference of greater than 100)

        desired_position_float: np.ndarray = self.get_input_port().Eval(context) # type: ignore

        # desired_position = np.clip(desired_position_float, -np.pi / 3, np.pi / 3)
        # TODO: add safety here for joint limits. ideally compute fk and then make sure we aren't hitting below z=0
        desired_position = (desired_position_float + np.pi) / (2 * np.pi) * 4096
        desired_position = desired_position.astype(int)

        measured_position = self._robot.read_position()
        current_pos = np.array(self._get_measured_pos_as_angles())

        # print(context)
        # robot_context = self.(context)
        # self._plant.SetPositions(robot_context, self._plant_robot, current_pos)
        
        # import ipdb; ipdb.set_trace()
        should_move = False
        joint_safety_diffs = [
        #     # (max_diff, min_diff)
            25,
            25,
            100, # this motor is a bit sticky, so require a greater min
            25,
            25,
            25,
        ]
        for i in range(self._num_movable_joints):
            abs_diff = abs(measured_position[i] - desired_position[i])
            min_diff = joint_safety_diffs[i]
                
            if abs_diff > min_diff:
                # only command a new position if there is somewhere new to move to
                should_move = True

        if should_move:
            logger.info(f"Current pos {measured_position}, desired pos as float {desired_position_float}, desired_position {desired_position}")
            # you can't reuse the measured position list for the rest of the joints because it's an unstable system and will droop!!
            commanded_position = list(desired_position[:self._num_movable_joints]) + list(self._home_position[self._num_movable_joints:])
            self._robot.queue_goal_pos(commanded_position) # gripper is always open for now.
        outputs.SetFromVector(current_pos) # type: ignore


    def _get_measured_pos_as_angles(self):
        measured_position = self._robot.read_position()
        return np.array(measured_position) / 4096 * 2 * np.pi - np.pi

    def test_conn(self):
        pos = self.robot.read_position()
        if not pos:
            raise Exception("Could not connect to bot")

    @property
    def robot(self):
        return self._robot


real_robot = None

def build_real_robot(meshcat):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-2)
    # Note that we parse into both the plant and the scene_graph here.
    robot_model = Parser(plant, scene_graph).AddModelsFromUrl(
            "file:///Users/raghav/Documents/projects/robot_arm/onshape-to-robot-examples/low-cost-robot/robot-post-processed.urdf"
    )[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("link_0"))
    plant.Finalize()

    # Wire up the plant to be position controlled? This is annoying, I'd rather just rely on "real" dynamics and somehow be able to set the plant positions correctly.
    low_cost_robot = plant.GetModelInstanceByName("onshape")
    kp = [100] * plant.num_positions()
    ki = [1] * plant.num_positions()
    kd = [10] * plant.num_positions()
    static_controller = builder.AddSystem(InverseDynamicsController(plant, kp, ki, kd, False))
    # extend_measured_pos = builder.AddSystem(ZeroExtendVec(6, 12))
    extend_desired_pos = builder.AddSystem(ZeroExtendVec(6, 12))

    

    # Connect the ID output to the plant
    builder.Connect(
        static_controller.get_output_port_control(), 
        plant.get_actuation_input_port()
    )

    builder.Connect(
        extend_desired_pos.get_output_port(),
        static_controller.get_input_port(1)
    )

    builder.Connect(
        plant.get_state_output_port(low_cost_robot),
        static_controller.get_input_port_estimated_state(),
    )

    
    
    # build the leaf system that is the robot
    global real_robot
    real_robot = builder.AddSystem(LowCostRobot(plant))
    
    builder.Connect(real_robot.get_output_port(), extend_desired_pos.get_input_port()) # desired state
    # builder.Connect(real_robot.get_output_port(), extend_measured_pos.get_input_port()) # measured state
    
    builder.ExportInput(real_robot.get_input_port(), "desired_position")
    builder.ExportOutput(real_robot.get_output_port(), "measured_position")
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    diagram = builder.Build()
    diagram.set_name("low-cost-robot (real)")
    return diagram, plant
    


def ik_example(meshcat, real_robot, show_diagram, destination_pos=[0.09970705, 0.18974534, 0.06858289]):
    if real_robot:
        (robot_diagram, plant) = build_real_robot(meshcat)
    else:
        (robot_diagram, plant) = build_sim_robot(meshcat)

    
    builder = DiagramBuilder()
    builder.AddSystem(robot_diagram)


    ee_pose_computer = builder.AddSystem(ComputeEePosition(plant))
    V_G_source = builder.AddSystem(BasicVelocityPlanner())
    V_G_source.set_name("v_WG")

    pinv_controller = builder.AddSystem(PseudoInverseController(plant))
    integrator = builder.AddSystem(Integrator(6))

    # connect
    builder.Connect(V_G_source.GetOutputPort("desired_ee_velocity"), pinv_controller.GetInputPort("desired_ee_velocity"))
    builder.Connect(robot_diagram.GetOutputPort("measured_position"), pinv_controller.GetInputPort("measured_position"))
    builder.Connect(pinv_controller.GetOutputPort("desired_velocity"), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), robot_diagram.GetInputPort("desired_position"))

    # connect ee pose computer
    builder.Connect(robot_diagram.GetOutputPort("measured_position"), ee_pose_computer.GetInputPort("joint_position"))

    # connect velocity source
    builder.Connect( ee_pose_computer.GetOutputPort("ee_position"), V_G_source.GetInputPort("current_ee_position"))

    diagram = builder.Build()
    
    if show_diagram:
        RenderDiagram(diagram, max_depth=1)
        wait_for_confirmation()

    logger.info("Built diagram")
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    plant_context = plant.GetMyContextFromRoot(context)


    plant.SetPositions(
        plant_context,
        [0, -np.pi / 2, 0, 0, 0, 0], # This should roughly match the initial positions from _home_pos
    )

    integrator.set_integral_value(
        integrator.GetMyContextFromRoot(context),
        plant.GetPositions(plant_context, plant.GetModelInstanceByName("onshape"),
        ),
    )

    v_g_context = V_G_source.GetMyContextFromRoot(context)
    V_G_source.GetInputPort("desired_ee_position").FixValue(v_g_context, destination_pos)
    
    logger.info("Starting simulation")
    simulator.set_target_realtime_rate(1.0)
    meshcat.StartRecording()
    total_time = 100
    simulator.AdvanceTo(total_time)
    meshcat.PublishRecording()

    wait_for_confirmation()


@cli.command("run_ik")
@click.option("--real-robot", is_flag=True)
@click.option("--show-diagram", is_flag=True)
def run_robot(real_robot, show_diagram):
    meshcat = Meshcat(port=7002)
    ik_example(meshcat, real_robot=real_robot, show_diagram=show_diagram)

@cli.command("gripper_frame_viz")
def gripper_frame_viz():
    meshcat = Meshcat(port=7002)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-2)
    parser = Parser(plant, scene_graph)
    parser.SetAutoRenaming(True)

    gripper_frames = make_gripper_frames()

    for (key, pose) in gripper_frames:
        g = parser.AddModelsFromUrl(
            "file:///Users/raghav/Documents/projects/robot_arm/onshape-to-robot-examples/low-cost-robot/robot-gripper.urdf"
            )[0]
        # TODO: the frames are in a weird position relative to each other, why does this happen?
        # plant.WeldFrames(plant.GetFrameByName("link_5", g), plant.GetFrameByName("moving_side", g))
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("moving_side", g), pose.pos)

    plant.Finalize()
    meshcat.Delete()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    wait_for_confirmation()

@cli.command("initial_cond_viz")
def initial_cond_viz():
    meshcat = Meshcat(port=7002)

    builder = DiagramBuilder()
    (robot_diagram, plant) = build_sim_robot(meshcat)
    builder.AddSystem(robot_diagram)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    for position in [
            np.zeros(6),
            [0, -np.pi / 2, 0, 0, 0, 0],
            [np.pi / 6, -np.pi / 3, np.pi / 6, -np.pi / 6, 0, 0]
            ]:

        logger.info(f"Setting position to {position}")
        plant.SetPositions(
                plant_context,
                position,
            )
        diagram.ForcedPublish(context)
        plant_pose = plant.get_body_poses_output_port().Eval(plant_context)
        gripper_pose = plant_pose[7]

        logger.info(f"EE pos: {gripper_pose.translation()}")
        logger.info(f"EE rot: {gripper_pose.rotation()}")
        wait_for_confirmation()



@cli.command("gripper_traj_viz")
def gripper_traj_viz():
    meshcat = Meshcat(port=7002)

    waypoints = make_gripper_frames()
    traj = make_gripper_trajectory(waypoints)
    traj_p_G = traj.get_position_trajectory()
    p_G = traj_p_G.vector_values(traj_p_G.get_segment_times())
    plt.plot(traj_p_G.get_segment_times(), p_G.T)
    plt.legend(["x", "y", "z"])
    plt.title("p_G")
    plt.show()


    traj_R_G = traj.get_orientation_trajectory()
    R_G = traj_R_G.vector_values(traj_R_G.get_segment_times())
    plt.plot(traj_R_G.get_segment_times(), R_G.T)
    plt.legend(["qx", "qy", "qz", "qw"])
    plt.title("R_G")
    plt.show()

    meshcat.ResetRenderMode()
    meshcat.SetLine("p_G", p_G, 2.0, rgba=Rgba(1, 0.65, 0))

    wait_for_confirmation()


def wait_for_confirmation(text="Press any key to continue"):
    try:
        input(text)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    cli()
