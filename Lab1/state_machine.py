import random
import math
from constants import *


class FiniteStateMachine(object):
    """
    A finite state machine.
    """
    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def update(self, agent):
        self.state.check_transition(agent, self)
        self.state.execute(agent)


class State(object):
    """
    Abstract state class.
    """
    def __init__(self, state_name):
        """
        Creates a state.

        :param state_name: the name of the state.
        :type state_name: str
        """
        self.state_name = state_name

    def check_transition(self, agent, fsm):
        """
        Checks conditions and execute a state transition if needed.

        :param agent: the agent where this state is being executed on.
        :param fsm: finite state machine associated to this state.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")

    def execute(self, agent):
        """
        Executes the state logic.

        :param agent: the agent where this state is being executed on.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")


class MoveForwardState(State):
    def __init__(self):
        super().__init__("MoveForward")
        # [DONE] Todo: add initialization code
        self.t = 0

    def check_transition(self, agent, state_machine):
        # [DONE] Todo: add logic to check and execute state transition
        self.t += SAMPLE_TIME

        # MoveForward -> MoveInSpiral
        if self.t > MOVE_FORWARD_TIME:
            state_machine.change_state(MoveInSpiralState())
            return
        
        # MoveForward -> GoBack
        if agent.get_bumper_state():
            state_machine.change_state(GoBackState())

    def execute(self, agent):
        # [DONE] Todo: add execution logic
        agent.set_velocity(FORWARD_SPEED, 0)


class MoveInSpiralState(State):
    def __init__(self):
        super().__init__("MoveInSpiral")
        # [DONE] Todo: add initialization code
        self.t = 0
    
    def check_transition(self, agent, state_machine):
        # [DONE] Todo: add logic to check and execute state transition
        self.t += SAMPLE_TIME

        # MoveInSpiral -> MoveForward
        if self.t > MOVE_IN_SPIRAL_TIME:
            state_machine.change_state(MoveForwardState())
            return
        
        # MoveInSpiral -> GoBack
        if agent.get_bumper_state():
            state_machine.change_state(GoBackState())

    def execute(self, agent):
        # [DONE] Todo: add execution logic
        inst_ang_speed = FORWARD_SPEED / (INITIAL_RADIUS_SPIRAL + SPIRAL_FACTOR * self.t) # speed / inst. radius
        agent.set_velocity(FORWARD_SPEED, inst_ang_speed)


class GoBackState(State):
    def __init__(self):
        super().__init__("GoBack")
        # [DONE] Todo: add initialization code
        self.t = 0

    def check_transition(self, agent, state_machine):
        # [DONE] Todo: add logic to check and execute state transition
        self.t += SAMPLE_TIME
        
        # MoveBack -> Rotate
        if self.t > GO_BACK_TIME:
            state_machine.change_state(RotateState())
            return

    def execute(self, agent):
        # [DONE] Todo: add execution logic
        agent.set_velocity(BACKWARD_SPEED, 0)


class RotateState(State):
    def __init__(self):
        super().__init__("Rotate")
        # [DONE] Todo: add initialization code
        self.rotation_time = random.uniform(0.5, 3)
        self.t = 0

    def check_transition(self, agent, state_machine):
        # [DONE] Todo: add logic to check and execute state transition
        self.t += SAMPLE_TIME
        
        # Rotate -> MoveForward
        if self.t > self.rotation_time:
            state_machine.change_state(MoveForwardState())
    
    def execute(self, agent):
        # [DONE] Todo: add execution logic
        agent.set_velocity(0, ANGULAR_SPEED)
