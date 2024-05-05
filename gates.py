'''
Quantum gates that are used in the circuit, as subclasses of qml.Operator.
'''
import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp


class U4Gate(Operation):
    '''
    U4 gate, a 15-parameter generalized 2-qubit gate
    '''
    num_params = 1 #  for pennylane internals, see nparams for actual number of parameters
    nparams = 15 # number of parameters
    num_wires = 2
    grad_method = "A"
    name = "U4"

    def __init__(self, theta, wires):
        if len(theta) != 15:
            raise ValueError("U4gate expects 15 parameters")
        super().__init__(theta, wires)

    def decomposition(self):
        '''
        Returns the decomposition of the U4 gate, as described in https://doi.org/10.1103%2Fphysreva.69.032315
        '''
        theta = self.parameters[0]
        wires = self.wires
        return [
            qml.U3(theta[0], theta[1], theta[2], wires=wires[0]),
            qml.U3(theta[3], theta[4], theta[5], wires=wires[1]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.RY(theta[6], wires=wires[0]),
            qml.RZ(theta[7], wires=wires[1]),
            qml.CNOT(wires=[wires[1], wires[0]]),
            qml.RY(theta[8], wires=wires[0]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.U3(theta[9], theta[10], theta[11], wires=wires[0]),
            qml.U3(theta[12], theta[13], theta[14], wires=wires[1])
        ]
    

class MGate(Operation):
    '''
    M gate, a 4-parameter 3-qubit gate used in the decomposition of the U8 gate.
    '''
    num_params = 1 #  for pennylane internals, see nparams for actual number of parameters
    nparams = 4 # number of parameters
    num_wires = 3
    grad_method = "A"
    name = "M gate"

    def __init__(self, theta, wires):
        if len(theta) != 4:
            raise ValueError("MGate expects 4 parameters")
        super().__init__(theta, wires)
    
    def decomposition(self):
        theta = self.parameters[0]
        wires = self.wires
        yield from [
            qml.CNOT(wires=wires[2::-2]),
            qml.CNOT(wires=wires[:2]),
            qml.CNOT(wires=wires[2:0:-1]),
            qml.RZ(-pnp.pi/2, wires=wires[2]),
            qml.RY(2*theta[0], wires=wires[2]),
            qml.CNOT(wires=wires[1:]),
            qml.RY(-2*theta[1], wires=wires[2]),
            qml.CNOT(wires=wires[1:]),
            qml.RZ(pnp.pi/2, wires=wires[2]),
            qml.CNOT(wires=wires[2::-2]),
            qml.CNOT(wires=wires[:2]),
            qml.CNOT(wires=wires[1::-1]),
            qml.Hadamard(wires=wires[2]),
            qml.CNOT(wires=wires[::2]),
            qml.RZ(2*theta[2], wires=wires[2]),
            qml.CNOT(wires=wires[::2]),
            qml.RZ(2 * theta[3], wires=wires[2]),
            qml.CNOT(wires=wires[1::-1]),
            qml.Hadamard(wires=wires[2])
        ]

class NGate(Operation):
    '''
    N gate, a 3-parameter 3-qubit gate used in the decomposition of the U8 gate.
    '''

    num_params = 1 #  for pennylane internals, see nparams for actual number of parameters
    nparams = 3 # number of parameters
    num_wires = 3
    grad_method = "A"
    name = "N gate"

    def __init__(self, theta, wires):
        if len(theta) != 3:
            raise ValueError("NGate expects 3 parameters")
        super().__init__(theta, wires)

    def decomposition(self):
        theta = self.parameters[0]
        wires = self.wires
        yield from [
            qml.RZ(-pnp.pi/2, wires=wires[1]),
            qml.Hadamard(wires=wires[2]),
            qml.CNOT(wires=wires[1::-1]),
            qml.CNOT(wires=wires[1:]),
            qml.RY(2*theta[0], wires=wires[1]),
            qml.CNOT(wires=wires[:2]),
            qml.RY(-2*theta[1], wires=wires[1]),
            qml.CNOT(wires=wires[:2]),
            qml.CNOT(wires=wires[1:]),
            qml.Hadamard(wires=wires[2]),
            qml.RZ(pnp.pi/2, wires=wires[1]),
            qml.CNOT(wires=wires[1::-1]),
            qml.CNOT(wires=wires[::2]),
            qml.CNOT(wires=wires[1:]),
            qml.RZ(2*theta[2], wires=wires[2]),
            qml.CNOT(wires=wires[1:]),
            qml.CNOT(wires=wires[::2])
        ]

class U8Gate(Operation):
    '''
    U8 gate, an 82-parameter generalized 3-qubit gate.
    '''
    num_params = 1 #  for pennylane internals, see nparams for actual number of parameters
    nparams = 82 # number of parameters
    num_wires = 3
    grad_method = "A"
    name = "U8"

    def __init__(self, theta, wires):
        if len(theta) != 82:
            raise ValueError("U8gate expects 82 parameters")
        super().__init__(theta, wires)

   
    def decomposition(self):
        '''
        Returns the decomposition of the U8 gate, as described in https://doi.org/10.48550/arXiv.quant-ph/0401178
        '''
        theta = self.parameters[0]
        wires = self.wires
        yield from [
            U4Gate(theta[:15], wires=wires[:2]),
            qml.U3(theta[15], theta[16], theta[17], wires=wires[2]),
            NGate(theta[18:21], wires=wires),
            U4Gate(theta[21:36], wires=wires[:2]),
            qml.U3(theta[36], theta[37], theta[38], wires=wires[2]),
            MGate(theta[39:43], wires=wires),
            U4Gate(theta[43:58], wires=wires[:2]),
            qml.U3(theta[58], theta[59], theta[60], wires=wires[2]),
            NGate(theta[61:64], wires=wires),
            U4Gate(theta[64:79], wires=wires[:2]),
            qml.U3(theta[79], theta[80], theta[81], wires=wires[2])
        ]
    
class PoolGate(Operation):
    '''
    Pooling gate that pools 2 wires into 1 wire. Accepts a pooling type as a hyperparameter.
    '''
    num_params = 1 #  for pennylane internals, see nparams for actual number of parameters
    num_wires = 2
    grad_method = "A"
    
    @property
    def name(self):
        return f"Pooling ({self.pool_type})"

    @classmethod
    def nparams(cls, pool_type):
        if pool_type == "trace":
            return 2
        if pool_type == "measure":
            return 3
        raise ValueError("pool_type must be 'trace' or 'measure'")

    def __init__(self, params, wires, pool_type="trace"):
        if pool_type not in ["trace", "measure"]:
            raise ValueError("pool_type must be 'trace' or 'measure'")
        if len(params) != self.nparams(pool_type):
            raise ValueError(f"pool_type {pool_type} expects {self.nparams(pool_type)} parameters, got {len(params)}")
        self.pool_type = pool_type
        super().__init__(params, wires)

    def decomposition(self):
        '''
        Decomposes pool gate, tracing out a wire or measuring and applying a conditional transformation.
        Wire 1 is pooled into wire 0.
        '''
        params = self.parameters[0]
        wires = self.wires
        if self.pool_type == "trace":
            return [
                qml.CRZ(params[0], wires=[wires[1], wires[0]]),
                qml.PauliX(wires=wires[1]),
                qml.CRX(params[1], wires=[wires[1], wires[0]])
            ]
        if self.pool_type == "measure":
            return qml.cond(qml.measure(wires=wires[1]), qml.U3)(params[0], params[1], params[2], wires=wires[0])
            
        
    
# class MultiCtrl(Operation):
#     '''
#     A multi-controlled version of the given operation with variable number of controls. The control is on
#      0 or 1 on each controlling qubit, given by hyperparameter 'ctrlstring', a bitstring of 0s and 1s.
#      deprecated in favor of qml.ctrl
#     '''

#     num_wires = qml.operation.AnyWires
#     grad_method = "A"
#     name = "MultiCtrl"

#     def label(self, decimals=None, base_label=None, cache=None):
#         return self.ctrlstring

#     def __init__(self, operation, ctrlstring, ctrlwires):
#         # Check that ctrlstring is a bitstring of 0s and 1s
#         self.operation = operation
#         self.ctrlstring = ctrlstring
#         self.ctrlwires = ctrlwires
#         if self.ctrlstring is not None and not all([i in ["0", "1"] for i in self.ctrlstring]):
#             raise ValueError(f"ctrlstring must be a bitstring of 0s and 1s, but got {self.ctrlstring}")
#                 # Check that ctrlstring and ctrlwires have the same length
#         if len(ctrlstring) != len(self.ctrlwires):
#             raise ValueError(f"ctrlstring and ctrlwires must have the same length, but got lengths {len(ctrlstring)}\
#                               and {len(self.ctrlwires)}")
#         super().__init__(wires=ctrlwires)
    
#     def decomposition(self):

#         # If ctrlstring is None, set it to a string of 1s of the same length as ctrlwires
#         ctrlstring = self.ctrlstring if self.ctrlstring is not None else '1'*len(self.wires)

#         # Check that ctrlwires is a unique list of wires
#         if len(self.wires) != len(set(self.wires)):
#             raise ValueError("ctrlwires must be a unique list of wires")

#         # Convert ctrlstring to a list of booleans
#         control_values = [bool(int(i)) for i in ctrlstring]

#         # Return a multi-controlled version of the operation
#         return qml.ctrl(self.operation, control=self.wires, control_values=control_values)