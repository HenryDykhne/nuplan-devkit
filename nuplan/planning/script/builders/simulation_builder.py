import copy
import logging
import os
import pickle
from typing import List, Optional, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractModification, AbstractScenarioModifier
from nuplan.planning.script.builders.metric_builder import build_metrics_engines
from nuplan.planning.script.builders.observation_builder import build_observations
from nuplan.planning.script.builders.occlusion_manager_builder import build_occlusion_manager
from nuplan.planning.script.builders.planner_builder import build_planners
from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.occlusion.abstract_occlusion_manager import AbstractOcclusionManager
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool

from nuplan.planning.script.builders.scenario_modifier_builder import build_scenario_modifiers

from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_simulations(
    cfg: DictConfig,
    worker: WorkerPool,
    callbacks: List[AbstractCallback],
    callbacks_worker: Optional[WorkerPool] = None,
    pre_built_planners: Optional[List[AbstractPlanner]] = None,
) -> List[SimulationRunner]:
    """
    Build simulations.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param callbacks: Callbacks for simulation.
    :param worker: Worker for job execution.
    :param callbacks_worker: worker pool to use for callbacks from sim
    :param pre_built_planners: List of pre-built planners to run in simulation.
    :return A dict of simulation engines with challenge names.
    """
    logger.info('Building simulations...')

    # Create Simulation object container
    simulations = list()

    # Retrieve scenarios
    logger.info('Extracting scenarios...')

    # Only allow simulation with NuPlanScenarioBuilder except when the NUPLAN_SIMULATION_ALLOW_ANY_BUILDER environment variable is set to a non-zero value.
    if not int(os.environ.get("NUPLAN_SIMULATION_ALLOW_ANY_BUILDER", "0")) and not is_target_type(
        cfg.scenario_builder, NuPlanScenarioBuilder
    ):
        raise ValueError(f"Simulation framework only runs with NuPlanScenarioBuilder. Got {cfg.scenario_builder}")

    scenario_filter = DistributedScenarioFilter(
        cfg=cfg,
        worker=worker,
        node_rank=int(os.environ.get("NODE_RANK", 0)),
        num_nodes=int(os.environ.get("NUM_NODES", 1)),
        synchronization_path=cfg.output_dir,
        timeout_seconds=cfg.distributed_timeout_seconds,
        distributed_mode=DistributedMode[cfg.distributed_mode],
    )
    scenarios = scenario_filter.get_scenarios()

    metric_engines_map = {}
    if cfg.run_metric:
        logger.info('Building metric engines...')
        metric_engines_map = build_metrics_engines(cfg=cfg, scenarios=scenarios)
        logger.info('Building metric engines...DONE')
    else:
        logger.info('Metric engine is disable')

    logger.info('Building simulations from %d scenarios...', len(scenarios))

    # Build a metric metadata file
    for scenario in scenarios:

        # Build planners
        if pre_built_planners is None:
            if 'planner' not in cfg.keys():
                raise KeyError('Planner not specified in config. Please specify a planner using "planner" field.')

            planners = build_planners(cfg.planner, scenario)
        else:
            planners = pre_built_planners

        for planner in planners:
            # Ego Controller
            ego_controller: AbstractEgoController = instantiate(cfg.ego_controller, scenario=scenario)

            # Simulation Manager
            simulation_time_controller: AbstractSimulationTimeController = instantiate(
                cfg.simulation_time_controller, scenario=scenario
            )

            # Perception
            observations: AbstractObservation = build_observations(cfg.observation, scenario=scenario)

            # Occlusions
            if 'occlusion_cfg' in cfg.keys() and cfg.occlusion_cfg.occlusion:
                occlusion_manager: AbstractOcclusionManager = build_occlusion_manager(cfg.occlusion_cfg, scenario=scenario)
            else:
                occlusion_manager = None

            # Metric Engine
            metric_engine = metric_engines_map.get(scenario.scenario_type, None)
            if metric_engine is not None:
                stateful_callbacks = [MetricCallback(metric_engine=metric_engine, worker_pool=callbacks_worker)]
            else:
                stateful_callbacks = []

            if "simulation_log_callback" in cfg.callback:
                stateful_callbacks.append(
                    instantiate(cfg.callback["simulation_log_callback"], worker_pool=callbacks_worker)
                )

            # Construct simulation and manager
            simulation_setup = SimulationSetup(
                time_controller=simulation_time_controller,
                observations=observations,
                ego_controller=ego_controller,
                occlusion_manager=occlusion_manager,
                scenario=scenario,
            )

            simulation = Simulation(
                simulation_setup=simulation_setup,
                callback=MultiCallback(callbacks + stateful_callbacks),
                simulation_history_buffer_duration=cfg.simulation_history_buffer_duration,
            )
            simulations.append(SimulationRunner(simulation, planner))
            
    # here we need to convert those simulations to our special scenarios
    if 'modify_scenario_simulations' in cfg and cfg.modify_scenario_simulations:
        logger.info('Modyfing Scenarios...')
        modification_file_path = 'modifications_for_second_testing_round.pkl'
        offshoot_scenario_simulations = []
        original_modified_tokens = []
        num_modifiable = 0
        original_num_runners = len(simulations)
        if 'second_testing_round' in cfg and cfg.second_testing_round:
            if 'modification_file_path' in cfg:
                modification_file_path = cfg.modification_file_path
            else:
                # we need to reload the modifications from the first round of testing
                assert 'scenarios_to_check' in cfg, 'You need to specify the scenario tokens to check in the alternate regime'
            with open(modification_file_path, 'rb') as f:
                modifications_for_second_testing_round = pickle.load(f)
                for sim in simulations:
                    if sim.simulation.scenario.token in modifications_for_second_testing_round:
                        num_modifiable += 1
                        for mod in modifications_for_second_testing_round[sim.simulation.scenario.token]:
                            if 'scenarios_to_check' not in cfg or sim.simulation.scenario.token + mod.modifier_string in cfg.scenarios_to_check:
                                clone = copy.deepcopy(sim)
                                clone.simulation.modification = mod
                                clone.scenario._modifier = mod.modifier_string
                                offshoot_scenario_simulations.append(clone)
                                original_modified_tokens.append(sim.simulation.scenario.token)
                original_modified_tokens = list(dict.fromkeys(original_modified_tokens)) #deduplicate
        else:
            num_gpus = cfg.number_of_gpus_allocated_per_simulation
            num_cpus = cfg.number_of_cpus_allocated_per_simulation
            print(num_cpus, num_gpus, 'are the number of cpus and gpus')
            offshoot_scenario_modifications: List[Tuple[str, str, List[AbstractModification]]] = worker.map(
                Task(fn=modify_simulations, num_gpus=num_gpus, num_cpus=num_cpus), simulations, [cfg]*len(simulations), verbose=True)
            modifications_for_second_testing_round = dict() #we need this to store all modifications for the second round of testing when we want to compare with and without occlusions
            for mods_for_sim, sim in tqdm(zip(offshoot_scenario_modifications, simulations)):
                if len(mods_for_sim[2]) > 0:
                    num_modifiable += 1
                    if len(simulations) < 500: #temporary measure to deal with jupyter notebooks truncating output
                        logger.info(mods_for_sim[1])
                    for mod in tqdm(mods_for_sim[2]):
                        clone = copy.deepcopy(sim)
                        clone.simulation.modification = mod
                        clone.scenario._modifier = mod.modifier_string
                        offshoot_scenario_simulations.append(clone)
                    modifications_for_second_testing_round[mods_for_sim[0]] = mods_for_sim[2]
                    original_modified_tokens.append(mods_for_sim[0])
            
            
            with open(modification_file_path, 'wb') as f:
                # saving the modifcations here so we can reload them later
                pickle.dump(modifications_for_second_testing_round, f)
            
            
        simulations = offshoot_scenario_simulations
        # you NEED to reset or reload the simulation for the modifications to take effect
        if len(simulations) < 500:
            print("[\n\t'"+("',\n\t'".join(original_modified_tokens))+"'\n]")
        logger.info(f'Created {len(simulations)} modified scenarios from {original_num_runners} scenarios, {num_modifiable} of which were modifiable.')
    logger.info('Building simulations...DONE!')
    return simulations

def modify_simulations(simulation: SimulationRunner, cfg: DictConfig) -> Tuple[str, str, List[AbstractModification]]:
    """_summary_
    :param simulation: _description_
    :param cfg: _description_
    :return: original simulation token, log message, and list of modifications
    """
    modifier_types = cfg.modifier_types
    scenario_modifiers = build_scenario_modifiers(modifier_types, cfg)
    all_modified_simulations = []
    log = ''
    for modifier in scenario_modifiers:
        modified_simulations = modifier.modify_scenario(simulation)
        log += f'Created {len(modified_simulations)} modified scenarios from scenario with token: {simulation.scenario.token}.\n'
    all_modified_simulations.extend(modified_simulations)
    log = log[:-1]
    
    return simulation.scenario.token, log, [new.simulation.modification for new in all_modified_simulations]
