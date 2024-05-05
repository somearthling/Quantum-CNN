'''
Sets the architecture of the convolutional, pooling, and densely-connected layers.
'''

import pennylane as qml
from pennylane.operation import Operation
import numpy as np
import itertools

import gates

class ConvParams:
    def __init__(self, filters=1, triple_parameters=False):
        self.filters = filters
        self.triple_parameters = triple_parameters

    def copy(self):
        return ConvParams(self.filters, self.triple_parameters)



class ConvLayer(Operation):
    '''
    Convolutional layer, using gates.U8Gate as a building block.
    '''

    num_params = 1 #  for pennylane internals, see nparams for actual number of parameters
    num_wires = qml.operation.AnyWires
    grad_method = "A"
    name = "Convolutional Layer"

    def __init__(self, params, wires, conv_params):
        super().__init__(params, wires)

        # Check that filters is an integer and does not exceed 2^(len(wires)-3)
        assert isinstance(conv_params.filters, int), "filters must be an integer"
        assert conv_params.filters <= 2**(len(wires)-3), "filters cannot be more than 2^(len(wires)-3)"

        # Check that triple_parameters is in [0, 1]
        assert conv_params.triple_parameters in [0, 1], "triple_parameters must be either 0 or 1"
        
        self.filters = conv_params.filters
        self.triple_parameters = bool(conv_params.triple_parameters)

    @classmethod
    def nparams(cls, conv_params):
        '''
        Returns the number of parameters needed for the convolutional layer.
        '''
        # Check that filters is an integer and conv_type is either 0 or 1
        assert isinstance(conv_params.filters, int), "filters must be an integer"
        assert conv_params.triple_parameters in [0, 1], "triple_parameters must be either 0 or 1"

        # Return the number of parameters needed for the convolutional layer
        return gates.U8Gate.nparams * (3**conv_params.triple_parameters)
    

    def decomposition(self):
        params = self.parameters[0]
        wires = self.wires

        current = 0 # current parameter index

        # Loop over wires 0, 1, 2
        for i in range(3):
            # Loop over every 3rd wire till the last wire without repitition, starting from the i'th wire
            for start in wires[i::3]:
                # Append a U8 gate with the next nparams parameters on the next 3 wires to the decomposition
                yield gates.U8Gate(params[current:current + gates.U8Gate.nparams], wires=[wires[start],wires[(start + 1)%len(wires)],wires[(start + 2)%len(wires)]])
            # If triple_parameters is True, move the current index forward by the total number of parameters the inner loop took
            current += gates.U8Gate.nparams * self.triple_parameters

        # If triple_parameters is False, move the current index forward by the total number of parameters
        current += gates.U8Gate.nparams

        if current < len(params):
            raise ValueError("Too many parameters supplied to ConvLayer")


class PoolParams:
    def __init__(self, pool_type="trace", separate_parameters=0, double_pool=False):
        self.pool_type = pool_type
        self.separate_parameters = separate_parameters
        self.double_pool = double_pool
        
class PoolLayer(Operation):
    '''
    Pooling layer, using gates.PoolGate as a building block.
    '''

    num_params = 1 #  for pennylane internals, see nparams for actual number of parameters
    num_wires = qml.operation.AnyWires
    grad_method = "A"
    name = "Pooling Layer"

    def __init__(self, params, wires=None, pool_params=None):
        super().__init__(params, wires=wires)

        if pool_params is None:
            pool_params = PoolParams()

        # Check that pool_type is either "trace" or "measure"
        assert pool_params.pool_type in ["trace", "measure"], "pool_type must be either 'trace' or 'measure'"
        # Check that separate_parameters is in [0, 1, 2]
        assert pool_params.separate_parameters in [0, 1, 2], "separate_parameters must be either 0, 1, or 2"
        # Check that double_pool is a boolean
        assert pool_params.double_pool in [0, 1], "double_pool must be a boolean"
        # Check that double_pool is False if pool_type is "measure", ADD DOUBLE POOLING FOR MEASURE LATER
        assert not (pool_params.double_pool and pool_params.pool_type == "measure"), (
            "double_pool must be False if pool_type is 'measure'"
        )
        # Check that double_pool is True if separate_parameters is 1
        assert pool_params.double_pool or pool_params.separate_parameters != 1, (
            "double_pool must be True if separate_parameters is 1. Set separate_parameters to 0 or 2 instead."
        )

        self.pool_type = pool_params.pool_type
        self.separate_parameters = pool_params.separate_parameters
        self.double_pool = pool_params.double_pool


    @staticmethod
    def nparams(wires, pool_params=None):
        '''
        Returns the number of parameters needed for the pooling layer.
        '''
        if pool_params is None:
            pool_params = PoolParams()

        pool_type = pool_params.pool_type
        separate_parameters = pool_params.separate_parameters
        double_pool = pool_params.double_pool

        # Check that pool_type is either "trace" or "measure"
        assert pool_type in ["trace", "measure"], "pool_type must be either 'trace' or 'measure'"
        # Check that separate_parameters is 0 or 1
        assert separate_parameters in [0, 1, 2], "separate_parameters must be in [0, 1, 2]"
        # Check that double_pool is 0 or 1
        assert double_pool in [0, 1], "double_pool must be bool"
        # Check that double_pool is False if pool_type is "measure", ADD DOUBLE POOLING FOR MEASURE LATER
        assert not (double_pool and pool_type == "measure"), (
            "double_pool must be False if pool_type is 'measure'"
        )
        # Check that double_pool is True if separate_parameters is 1
        assert double_pool or separate_parameters != 1, (
            "double_pool must be True if separate_parameters is 1. Set separate_parameters to 0 or 2 instead."
        )

        # Return the number of parameters needed for the pooling layer
        match pool_params.separate_parameters:
            case 0:
                distinct_gates = 1
            case 1:
                distinct_gates = 2
            case 2:
                distinct_gates = ((len(wires) // 2)*(double_pool + 1))+(len(wires) % 2)

        pool_params = gates.PoolGate.nparams(pool_params.pool_type)

        return distinct_gates * pool_params

    def decomposition(self):
        params = self.parameters[0]
        wires = self.wires

        current = 0 # current parameter index

        # Loop over every other wire starting from the 0th wire
        for i in range(len(wires))[::2]:
            # Append a PoolGate with the next nparams parameters and the next 2 wires to the decomposition
            yield gates.PoolGate(
                params[current:current + gates.PoolGate.nparams(self.pool_type)], 
                wires=[wires[i], wires[(i + 1)%len(wires)]], 
                pool_type=self.pool_type
            )
            # Move the current index by nparams if separate_parameters is 2
            current += gates.PoolGate.nparams(self.pool_type) * (self.separate_parameters == 2)

        # Move the current index by nparams if separate_parameters is 1
        current += gates.PoolGate.nparams(self.pool_type) * (self.separate_parameters == 1)

        # Loop over every other wire starting from the 1st wire if double_pool is True
        if self.double_pool:
            for i in range(1, len(wires))[::2]:
                # Append a PoolGate with the next nparams parameters and the next 2 wires to the decomposition
                yield gates.PoolGate(
                    params[current:current + gates.PoolGate.nparams(self.pool_type)], 
                    wires=[wires[(i + 1)%len(wires)], wires[i]],
                    pool_type=self.pool_type
                )
                # Move the current index by nparams if separate_parameters is 2
                current += gates.PoolGate.nparams(self.pool_type) * (self.separate_parameters == 2)

        # Move the current index by nparams if separate_parameters is 0 or 1
        current += gates.PoolGate.nparams(self.pool_type) * (self.separate_parameters in [0, 1])

        if current < len(params):
            raise ValueError("Too many parameters supplied to PoolLayer")

class DenseParams:
    def __init__(self, structurestring='ii ii', read=None, listranges=None, imprimitives=None, pool_type="trace"):
        self.structurestring = structurestring
        self.read = read
        self.listranges = listranges if listranges is not None else [None]*structurestring.count('s')
        self.imprimitives = imprimitives if imprimitives is not None else [None]*structurestring.count('s')
        self.pool_type = pool_type


class DenseLayer(Operation):
    '''
    Dense layer, using gates.U4Gate, gates.PoolGate and qml.StronglyEntanglingLayers as building blocks. 
    Takes a 'structurestring' as a hyperparameter to specify the structure of the dense layer.
    'p' for pooling, 'i' for independent U4 gates, 'r' for random U4 gates, 
    'b' to use the qml.broadcast function with all-to-all pattern, 's' for strongly entangling gates.
    A space can be used ONCE to reduce the active wires to the first 'read' wires.
    Hyperparameters layers, ranges and imprimitives further specify the structure of the strongly entangling gates.
    ranges and imprimitives must be lists of same length as the number of strongly entangling sublayers.
    ranges must be a list of lists of natural numbers that specifies the ranges parameter for the strongly entangling layers.
    imprimitives must be a list of subclasses of qml.Operation.
    read specifies the number of output wires. pool_type specifies the type of pooling to use.
    '''

    num_params = 1 #  for pennylane internals, see nparams for actual number of parameters
    num_wires = qml.operation.AnyWires
    grad_method = "A"
    name = "Dense Layer"

    def __init__(self, params, wires, dense_params):
        super().__init__(params, wires)

        # Check that structurestring is a string
        assert isinstance(dense_params.structurestring, str), "structurestring must be a string"
        # Check that structurestring only contains characters 'p', 'i', 'r', 'b', 's' and ' '
        assert all(char in ['p', 'i', 'r', 'b', 's', ' '] for char in dense_params.structurestring), (
            "structurestring must only contain 'p', 'i', 'r', 'b', 's' and at most one space"
        )
        # Check that structurestring contains at most one space
        assert dense_params.structurestring.count(' ') <= 1, "structurestring must contain at most one space"

        # Check that read is an integer
        assert isinstance(dense_params.read, int), "read must be an integer"
        # Check that read is less than or equal to the number of wires
        assert dense_params.read <= len(wires), "read must be less than or equal to the number of wires"

        # Check that listranges is a list
        assert isinstance(dense_params.listranges, list), "listranges must be a list"
        # Check that each element of listranges is None or a list of natural numbers
        assert all(
            ranges is None or (
                isinstance(ranges, list) and all(
                    isinstance(range_, int) and range_ > 0 for range_ in ranges
                )
            ) for ranges in dense_params.listranges
        ), "listranges must a list of Nones or a list of sequences of natural numbers"
        # Check that len(listranges) is equal to the number of occurences of 's' in structurestring
        assert dense_params.structurestring.count('s') == len(dense_params.listranges), (
            "len(listranges) must be equal to the number of occurences of 's' in structurestring"
        )

        # Check that imprimitives is list
        assert isinstance(dense_params.imprimitives, list), "imprimitives must be a list"
        # Check that each element of imprimitives is None or a subclass of qml.Operation that acts on 2 wires and takes no parameters
        assert all(
            imprimitive is None or (
                issubclass(imprimitive, Operation) and imprimitive.num_wires == 2 and imprimitive.num_params == 0
            ) for imprimitive in dense_params.imprimitives
        ), (
            "imprimitives must be a list of Nones or a list of subclasses of qml.Operation that act on 2 wires and take no parameters"
        )
        # Check that len(imprimitives) is equal to the number of occurences of 's' in structurestring
        assert dense_params.structurestring.count('s') == len(dense_params.imprimitives), (
            "len(imprimitives) must be equal to the number of occurences of 's' in structurestring"
        )

        # Check that pool_type is either "trace" or "measure"
        assert dense_params.pool_type in ["trace", "measure"], "pool_type must be either 'trace' or 'measure'"

        self.structurestring = dense_params.structurestring
        self.read = dense_params.read
        self.ranges = dense_params.listranges
        self.imprimitives = dense_params.imprimitives
        self.pool_type = dense_params.pool_type

        self.layers = [len(ranges) if ranges is not None else 1 for ranges in dense_params.listranges]

    @staticmethod
    def _build_pool_sublayer(params, wires, out_wires, pool_type):
        '''
        Builds PoolGates to pool each wire not in out_wires wires onto out_wires.
        '''
        # Identify elements of wires not in out_wires
        in_wires = [wire for wire in wires if wire not in out_wires]
        
        current = 0 # current parameter index

        # Loop over each wire in out_wires
        for out_wire in out_wires:
            # Loop over each wire in in_wires
            for wire in in_wires:
                # Append a PoolGate with the next nparams parameters and the next 2 wires to the decomposition
                yield gates.PoolGate(params[:gates.PoolGate.nparams(pool_type)], wires=[out_wire, wire], pool_type=pool_type)
                # Move the current index by nparams
                current += gates.PoolGate.nparams(pool_type)

    @staticmethod
    def _pool_sublayer_nparams(wires, out_wires, pool_type):
        '''
        Returns the number of parameters needed for the pool sublayer.
        '''
        return len(out_wires) * (len(wires)-len(out_wires)) * gates.PoolGate.nparams(pool_type)

    @staticmethod
    def _build_independent_sublayer(params, wires):
        '''
        Builds dense connection sublayer that maximise the number of independent U4 gates.
        '''
        current = 0

        # iterate over all possible distances between starting wires without repetition
        for i in range(1, len(wires)//2 +1):
            start_wires = []
            # if i is half the number of wires, only iterate the wires below half the wires, since
            # (a, a+n/2) and (a+n/2, a) are the same - this is only relevant for even numbers of wires
            if i == len(wires)/2:
                # starting wires form batches, each with start point j
                for j in range(0,min(i+1, len(wires)//2),2):
                    # starting wires are j, j+i+1, j+2i+2, ...
                    for k in range(j, len(wires), i+1):
                        start_wires.append(k)
            else:
                # starting wires form batches, each with start point j
                for j in range(i+1):
                    # starting wires are j, j+i+1, j+2i+2, ...
                    for k in range(j, len(wires), i+1):
                        start_wires.append(k)
            # construct U4 gates between the start_wires and start_wires+i
            for j in start_wires:
                # append a U4 gate with the next nparams parameters across j and j+i
                yield gates.U4Gate(
                    params[current:current+gates.U4Gate.nparams], 
                    wires=[wires[j % len(wires)], wires[(j+i) % len(wires)]]
                )
                # move the current index by nparams
                current += gates.U4Gate.nparams

    @staticmethod
    def _build_random_sublayer(params, wires):
        '''
        Builds dense connection sublayer with randomly distributed U4 gates.
        '''
        current = 0

        # obtain list of all possible wire pairs in wires
        wire_pairs = []
        for idx, wire_1 in enumerate(wires):
            for wire_2 in wires[idx+1:]:
                wire_pairs.append([wire_1, wire_2])

        # randomly shuffle the wire pairs, seed is set in main and printed to results
        np.random.shuffle(wire_pairs)

        # construct U4 gates between the shuffled wire pairs
        for pair in wire_pairs:
            # append a U4 gate with the next nparams parameters across the pair
            yield gates.U4Gate(params[current:current+gates.U4Gate.nparams], wires=pair)
            # move the current index by nparams
            current += gates.U4Gate.nparams

    @staticmethod
    def _build_all_to_all_sublayer(params, wires):
        '''
        Builds dense connection sublayer with all-to-all connections like qml.broadcast.
        '''

        for i, start_wire in enumerate(wires):
            for end_wire in wires[i+1:]:
                yield gates.U4Gate(params[:gates.U4Gate.nparams], wires=[start_wire, end_wire])
    
    @staticmethod
    def _u4_sublayer_nparams(wires):
        '''
        Returns the number of parameters needed for the U4 sublayer.
        '''
        return int(len(wires) * (len(wires)-1) / 2 * gates.U4Gate.nparams)
        
    @staticmethod
    def _build_strongly_entangling_sublayer(params, wires, ranges=None, imprimitive=None):
        '''
        Builds dense connection sublayer like qml.StronglyEntanglingLayers.
        '''
        # check that len(params) is divisible by qml.U3.num_params * len(wires)
        assert len(params) % (qml.U3.num_params * len(wires)) == 0, "number of parameters must be divisible by 3 * len(wires)"

        current = 0 # current parameter index
        layers = len(params) // (len(wires) * qml.U3.num_params) # number of layers
        if layers >= len(wires):
            raise ValueError("Number of layers must be less than the number of wires")
        imprimitive = qml.CNOT if imprimitive is None else imprimitive # default imprimitive is qml.CNOT
        ranges = [layer % len(wires) for layer in range(1,layers+1)] if ranges is None else ranges # default ranges is [1, 2, ..., len(wires)-1]

        for range_ in ranges:
            # build single qubit rotation gates
            for wire in wires:
                yield qml.U3(*params[current:current+qml.U3.num_params], wires=wire)
                current += qml.U3.num_params
            # build imprimitive gates with range distance between wires
            for i, wire in enumerate(wires):
                yield imprimitive(wires=[wire, wires[(i+range_) % len(wires)]])


    @staticmethod
    def _strongly_entangling_sublayer_nparams(wires, layers):
        '''
        Returns the number of parameters needed for the strongly entangling sublayer.
        '''
        return layers * len(wires) * 3

    def decomposition(self):
        '''
        Iterates over the structurestring and builds the dense layer.
        '''
        params = self.parameters[0]
        wires = self.wires

        current = 0 # current parameter index
        s_counter = 0 # counter for the number of strongly entangling sublayers

        pool_nparams = self.__class__._pool_sublayer_nparams # pylint: disable=protected-access
        u4_nparams = self.__class__._u4_sublayer_nparams # pylint: disable=protected-access
        strongly_entangling_nparams = self.__class__._strongly_entangling_sublayer_nparams # pylint: disable=protected-access

        for char in self.structurestring:
            match char:
                case 'p':
                    yield from self._build_pool_sublayer(
                        params[current:current+pool_nparams(wires, wires[:self.read], self.pool_type)],
                        wires, wires[:self.read], self.pool_type
                    )
                    current += pool_nparams(wires, wires[:self.read], self.pool_type)
                case 'i':
                    yield from self._build_independent_sublayer(
                        params[current:current+u4_nparams(wires)],
                        wires
                    )
                    current += u4_nparams(wires)
                case 'r':
                    yield from self._build_random_sublayer(
                        params[current:current+u4_nparams(wires)],
                        wires
                    )
                    current += u4_nparams(wires)
                case 'b':
                    yield from self._build_all_to_all_sublayer(
                        params[current:current+u4_nparams(wires)],
                        wires
                    )
                    current += u4_nparams(wires)
                case 's':
                    if self.ranges is not None and self.imprimitives is not None:
                        yield from self._build_strongly_entangling_sublayer(
                            params[current:current+strongly_entangling_nparams(wires, self.layers[s_counter])],
                            wires, ranges=self.ranges[s_counter], imprimitive=self.imprimitives[s_counter]
                        )
                    else:
                        yield from self._build_strongly_entangling_sublayer(
                            params[current:current+strongly_entangling_nparams(wires, self.layers[s_counter])],
                            wires
                        )
                    current += strongly_entangling_nparams(wires, self.layers[s_counter])
                    s_counter += 1
                case ' ':
                    wires = wires[:self.read]

        if current < len(params):
            raise ValueError("Too many parameters supplied to DenseLayer")
    
    @classmethod
    def nparams(cls, wires, dense_params):
        '''
        Returns the number of parameters needed for the dense layer.
        '''
        structurestring = dense_params.structurestring
        read = dense_params.read
        listranges = dense_params.listranges
        pool_type = dense_params.pool_type

        # Check that structurestring is a string
        assert isinstance(structurestring, str), "structurestring must be a string"
        # Check that structurestring only contains characters 'p', 'i', 'r', 'b', 's' and ' '
        assert all(char in ['p', 'i', 'r', 'b', 's', ' '] for char in structurestring), (
            "structurestring must only contain 'p', 'i', 'r', 'b', 's' and at most one space"
        )
        # Check that structurestring contains at most one space
        assert structurestring.count(' ') <= 1, "structurestring must contain at most one space"

        # Check that read is an integer
        assert isinstance(read, int), "read must be an integer"
        # Check that read is less than or equal to the number of wires
        assert read <= len(wires), "read must be less than or equal to the number of wires"

      # Check that listranges is a list
        assert isinstance(dense_params.listranges, list), "listranges must be a list"
        # Check that each element of listranges is None or a list of natural numbers
        assert all(
            ranges is None or (
                isinstance(ranges, list) and all(
                    isinstance(range_, int) and range_ > 0 for range_ in ranges
                )
            ) for ranges in dense_params.listranges
        ), "listranges must a list of Nones or a list of sequences of natural numbers"
        # Check that len(listranges) is equal to the number of occurences of 's' in structurestring
        assert dense_params.structurestring.count('s') == len(dense_params.listranges), (
            "len(listranges) must be equal to the number of occurences of 's' in structurestring"
        )
        # Check that len(listranges) is equal to the number of occurences of 's' in structurestring
        assert listranges is None or structurestring.count('s') == len(listranges), (
            "len(listranges) must be equal to the number of occurences of 's' in structurestring"
        )

        layers = [len(ranges) if ranges is not None else 1 for ranges in listranges]

        # Check that pool_type is either "trace" or "measure"
        assert pool_type in ["trace", "measure"], "pool_type must be either 'trace' or 'measure'"

        nparams = 0 # number of parameters needed for the dense layer
        s_counter = 0 # counter for the number of strongly entangling sublayers
        # Loop over the structurestring
        for char in structurestring:
            match char:
                case 'p':
                    nparams += cls._pool_sublayer_nparams(wires, wires[:read], pool_type)
                case 'i' | 'r' | 'b':
                    nparams += cls._u4_sublayer_nparams(wires)
                case 's':
                    nparams += cls._strongly_entangling_sublayer_nparams(wires, layers[s_counter])
                    s_counter += 1
                case ' ':
                    wires = wires[:read]

        return nparams
    

class QCNN(Operation):
    '''
    Quantum convolutional neural network, using ConvLayer, PoolLayer and DenseLayer as building blocks.
    '''

    num_params = 1 #  for pennylane internals, see nparams for actual number of parameters
    num_wires = qml.operation.AnyWires
    grad_method = "A"
    name = "QCNN"

    def __init__(
            self, params, wires, conv_params=None, pool_params=None, dense_params=None):
        
        if conv_params is None:
            conv_params = ConvParams()
        if pool_params is None:
            pool_params = PoolParams()
        if dense_params is None:
            dense_params = DenseParams()
        
        super().__init__(params, wires)
        self.conv_params = conv_params
        self.pool_params = pool_params
        self.dense_params = dense_params

    def decomposition(self):
        params = self.parameters[0]
        active_wires = self.wires # track the wires that are currently active
        conv_params = ConvParams(filters=self.conv_params.filters, triple_parameters=self.conv_params.triple_parameters)
        current_parameter = 0 # current parameter index

        # Loop over the active_wires until there are less than 2 * self.dense_params.read wires so that the last layer is a DenseLayer
        while len(active_wires) > 2 * self.dense_params.read:            
            # Append a ConvLayer with the next Convlayer.nparams parameters on the active_wires
            yield ConvLayer(
                params[current_parameter:current_parameter + ConvLayer.nparams(
                    conv_params=conv_params
                )], wires=active_wires, conv_params=self.conv_params
            )
            # Move the current index by ConvLayer.nparams
            current_parameter += ConvLayer.nparams(self.conv_params)
            # Set filters to 1 after the first ConvLayer - relevant only if ConvLayer changes in the future
            conv_params = ConvParams(filters=1, triple_parameters=self.conv_params.triple_parameters)
            # Append a PoolLayer with the next PoolLayer.nparams parameters on the active_wires
            yield PoolLayer(
                params[current_parameter:current_parameter + PoolLayer.nparams(
                    active_wires, pool_params=self.pool_params
                )], wires=active_wires, pool_params=self.pool_params
            )
            # Move the current index by PoolLayer.nparams
            current_parameter += PoolLayer.nparams(
                active_wires, pool_params=self.pool_params
            )
            # Update active_wires
            active_wires = active_wires[::2]

        # Append a DenseLayer with the next DenseLayer.nparams parameters on the active_wires
        
        yield DenseLayer(
            params[current_parameter:current_parameter + DenseLayer.nparams(
                active_wires, dense_params=self.dense_params
            )], wires=active_wires, dense_params=self.dense_params
        )

        current_parameter += DenseLayer.nparams(active_wires, dense_params=self.dense_params)

        if current_parameter < len(params):
            raise ValueError("Too many parameters supplied to QCNN")


    @classmethod
    def nparams(cls, wires, conv_params=None, pool_params=None, dense_params=None):
        '''
        Returns the number of parameters needed for the QCNN.
        '''
        active_wires = wires
        nparams = 0 # number of parameters needed for the QCNN
        new_conv_params = ConvParams(filters=conv_params.filters, triple_parameters=conv_params.triple_parameters)

        while len(active_wires) > 2 * dense_params.read:
            nparams += ConvLayer.nparams(conv_params=conv_params)
            new_conv_params = ConvParams(filters=1, triple_parameters=conv_params.triple_parameters)
            nparams += PoolLayer.nparams(active_wires, pool_params=pool_params)
            active_wires = active_wires[::2]

        nparams += DenseLayer.nparams(active_wires, dense_params=dense_params)

        return nparams
    
    @classmethod
    def out_wires(cls, wires, conv_params, dense_params):
        '''
        Returns the list of output wires to read from
        '''

        active_wires = wires[:-int(np.ceil(np.log2(conv_params.filters))) or None]
        while len(active_wires) > 2 * dense_params.read:
            active_wires = active_wires[::2]

        return active_wires[:dense_params.read]