from dataclasses import dataclass
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix, PiecewisePose

@dataclass
class GripperFrame:
    pos: RigidTransform
    time: float


Waypoints = list[tuple[str, GripperFrame]]


def make_gripper_frames() -> Waypoints:
    return [
            ("start", GripperFrame(
                    RigidTransform(
                        RotationMatrix.MakeXRotation(0),
                        [ 0.01227231,  0.07128447, 0.06267743]
                    ),
                    0.0
                )),
            ("end", GripperFrame(
                    RigidTransform(
                        RotationMatrix.MakeXRotation(0),
                        [0.06299721,  0.20357181, 0.08171054]
                    ),
                    1.0
                )
             )
            ]

def make_gripper_trajectory(waypoints: Waypoints) -> PiecewisePose:
    # Make sure times are increasing
    assert all(waypoints[i][1].time < waypoints[i+1][1].time for i in range(len(waypoints) - 1))

    times = [waypoint[1].time for waypoint in waypoints]
    poses = [waypoint[1].pos for waypoint in waypoints]
    return PiecewisePose.MakeLinear(times, poses)
