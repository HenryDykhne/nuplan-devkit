from abc import ABCMeta, abstractmethod
from typing import List

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner

class AbstractScenarioModifier(metaclass=ABCMeta):
    @abstractmethod
    def modify_scenario(self, runner: SimulationRunner) -> List[SimulationRunner]:
        """We convert one abstract scenario into many abstract scenarios by modifying the scenario in some way.
        :param runner: a simualtion runner
        :return: we return a list of scenarios that are modified versions of the input scenario
        """
        pass