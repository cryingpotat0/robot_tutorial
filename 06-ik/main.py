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
    Simulator,
    InverseDynamicsController,
    LeafSystem,
    JointSliders,
    Meshcat,
    FirstOrderLowPassFilter,
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

logger = logging.getLogger("robot")
logging.basicConfig(level=logging.INFO)

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

        # Slowly move the robot to the home position
        self._home_position = [2048, 1024, 2048, 1709, 2067, 2367]
        self._num_movable_joints = num_movable_joints
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
            # (max_diff, min_diff)
            (300, 25),
            (300, 25),
            (400, 100), # this motor is a bit sticky, so require a greater min
            (300, 25),
            (300, 25),
            (300, 25),
        ]
        for i in range(self._num_movable_joints):
            abs_diff = abs(measured_position[i] - desired_position[i])
            max_diff, min_diff = joint_safety_diffs[i]
            if abs_diff > max_diff:
                print("Desired position too far from current position, skipping")
                print(f"Current pos {measured_position}, desired pos as float {desired_position_float}, desired_position {desired_position}")
                outputs.SetFromVector(current_pos) # type: ignore
                return
                
            elif abs_diff > min_diff:
                # only command a new position if there is somewhere new to move to
                should_move = True

        if should_move:
            print(f"Current pos {measured_position}, desired pos as float {desired_position_float}, desired_position {desired_position}")
            # you can't reuse the measured position list for the rest of the joints because it's an unstable system and will droop!!
            commanded_position = list(desired_position[:self._num_movable_joints]) + list(self._home_position[self._num_movable_joints:])
            print(commanded_position)
            # TODO: put this in a queue that the robot can read so that the main sim thread is not blocked.
            self._robot.set_goal_pos(commanded_position) # gripper is always open for now.
        outputs.SetFromVector(np.array(self._get_measured_pos_as_angles())) # type: ignore


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
    


def ik_example(meshcat, real_robot):
    if real_robot:
        (robot_diagram, plant) = build_real_robot(meshcat)
    else:
        (robot_diagram, plant) = build_sim_robot(meshcat)
    
    builder = DiagramBuilder()
    # teleop = builder.AddSystem(JointSliders(meshcat=meshcat, plant=plant))

    builder.AddSystem(robot_diagram)

    # connect jacobian controller
    pinv_controller = builder.AddSystem(PseudoInverseController(plant))
    integrator = builder.AddSystem(Integrator(6))
    # velocity = builder.AddSystem(ConstantValueSource(BasicVector([
    #             0,  # rotation about x
    #             0,  # rotation about y
    #             0,  # rotation about z
    #             0.3,  # x
    #             0,  # y
    #             0.3, # z
    #         ])))
    # builder.Connect(velocity.get_output_port(), pinv_controller.get_input_port("desired_ee_velocity"))

    builder.Connect(robot_diagram.get_output_port(0), pinv_controller.get_input_port(0))

    builder.Connect(pinv_controller.get_output_port(), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), robot_diagram.get_input_port(0))


    # you need a sink for teh sim to do anything SMFH.
    # log_sink = builder.AddSystem(VectorLogSink(6))
    # builder.Connect(robot_diagram.get_output_port(0), log_sink.get_input_port())

    diagram = builder.Build()
    
    RenderDiagram(diagram, max_depth=1)
    logger.info("Built diagram")
    simulator = Simulator(diagram)
    logger.info("Starting simulation")
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(1)


@click.command()
@click.option("--real-robot", is_flag=True)
def run_robot(real_robot):
    meshcat = Meshcat(port=7002)
    ik_example(meshcat, real_robot=real_robot)

if __name__ == "__main__":
    run_robot()
