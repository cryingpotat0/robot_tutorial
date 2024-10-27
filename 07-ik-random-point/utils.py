from pydot import pydot
from pydrake.all import LeafSystem
import numpy as np
import io
from typing import Optional
from pydrake.systems.framework import System
from PIL import Image


class TruncateVec(LeafSystem):
    def __init__(self, input_dims, output_dims):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("input", input_dims)
        self.DeclareVectorOutputPort("output", output_dims, self.CalcOutput)
        self.input_dims = input_dims
        self.output_dims = output_dims

    def CalcOutput(self, context, outputs):
        full_state = self.get_input_port().Eval(context)
        outputs.SetFromVector(full_state[:self.output_dims]) # TODO: support arbitrary slicing

class ZeroExtendVec(LeafSystem):
    def __init__(self, input_dims, output_dims):
        LeafSystem.__init__(self)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.DeclareVectorInputPort("input", input_dims)
        self.DeclareVectorOutputPort("output", output_dims, self.CalcOutput)

    def CalcOutput(self, context, outputs):
        input_vec = self.get_input_port(0).Eval(context)
        outputs.SetFromVector(np.hstack((input_vec, [0] * (self.output_dims - self.input_dims))))


class Identity(TruncateVec):
    def __init__(self, dims):
        TruncateVec.__init__(self, dims, dims)
        

def RenderDiagram(diagram: System, max_depth: Optional[int] = None):
    svg = pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=max_depth))[
            0
            ].create_png(prog='dot')
    # treat the DOT output as an image file
    sio = io.BytesIO()
    sio.write(svg)
    sio.seek(0)
    img = Image.open(sio)
    img.show()
