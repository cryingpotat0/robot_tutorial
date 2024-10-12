from pydrake.all import (
        JacobianWrtVariable,
        LeafSystem
        )

import numpy as np


class PseudoInverseController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._robot = plant.GetModelInstanceByName("onshape")
        self._G = plant.GetBodyByName("link_5").body_frame()
        self._W = plant.world_frame()

        self.DeclareVectorInputPort("measured_position", 6) # type: ignore
        # self.DeclareVectorInputPort("desired_ee_velocity", 6) # type: ignore
        self.DeclareVectorOutputPort("desired_velocity", 6, self.CalcOutput) # type: ignore

    def CalcOutput(self, context, outputs):
        q = self.get_input_port().Eval(context)
        # V_G_desired = self.get_input_port(1).Eval(context)
        self._plant.SetPositions(self._plant_context, self._robot, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )
        # J_G = J_G[:, 0:]  # Ignore gripper terms
        V_G_desired = np.array(
            [
                0,  # rotation about x
                0,  # rotation about y
                0,  # rotation about z
                0.3,  # x
                0,  # y
                0.3, # z
            ]
        ) 
        v = np.linalg.pinv(J_G).dot(V_G_desired)
        outputs.SetFromVector(v) # type: ignore

