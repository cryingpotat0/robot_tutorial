from dataclasses import dataclass
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix, PiecewisePose

@dataclass
class GripperFrame:
    pos: RigidTransform
    time: float


Waypoints = list[tuple[str, GripperFrame]]


def make_gripper_frames() -> Waypoints:
    # Get these by running initial_cond_viz until you have reasonable points.
    return [
            ("start", GripperFrame(
                    RigidTransform(
                        RotationMatrix(
                            [
                             [3.528098644890758e-14, -1.4645942698561294e-13, -1.0],
                             [0.23832541480209682, 0.9711853564893822, -1.3383089507819007e-13],
                             [0.9711853564893822, -0.23832541480209682, 6.916938108970298e-14], 
                            ],
                        ),
                        [0.01227231, 0.13794192, 0.08648411]
                    ),
                    0.0
                )),
            ("end", GripperFrame(
                    RigidTransform(
                        RotationMatrix(
                            [
                                [-0.13959840732927445, 0.4801169489520393, -0.866025403784505],
                                [-0.24179153415007085, 0.8315869491601404, 0.4999999999998852],
                                [0.9602338979042996, 0.2791968146586111, -1.0937239345091733e-15],
                            ],
                        ),
                        [0.09970705, 0.18974534, 0.06858289]
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
