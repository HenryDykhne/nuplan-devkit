from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from nuplan.planning.simulation.simulation import Simulation
    from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner


class AbstractScenarioModifier(metaclass=ABCMeta):
    @abstractmethod
    def modify_scenario(self, runner: SimulationRunner) -> List[SimulationRunner]:
        """We convert one abstract scenario into many abstract scenarios by modifying the scenario in some way.
        :param runner: a simualtion runner
        :return: we return a list of simualtion runners that are modified versions of the input scenario
        """
        pass
    
    
class AbstractModification(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, modifier_string: str):
        self.modifier_string = modifier_string
    
    def modify(self, simulation: Simulation) -> None:
        """We perform the modification on the simulation. This helps with reimplementing the modification after the simulation has been unpickled.
        :param scenario: a simulation
        """
        pass