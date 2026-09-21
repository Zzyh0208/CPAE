import argparse
import json
from cape.configuration import Configuration
from cape.runtime.sumo import SUMORuntime
from cape.learning.agent import CAPEAgent
from cape.learning.trainer import TrainingPipeline

parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/cape.yaml'); parser.add_argument('--manifest', required=True); parser.add_argument('--output', default='artifacts/test'); arguments = parser.parse_args()
configuration = Configuration.load(arguments.config); network, model, control = configuration.section('network'), configuration.section('model'), configuration.section('control')
runtime = SUMORuntime(arguments.manifest, network['history'], control['period_seconds']); agent = CAPEAgent(network['features'], model['hidden'], model['code_dim'], control['measures'], model['heads'], configuration.values['device']); result = TrainingPipeline(runtime, agent, configuration).evaluate('test', arguments.output); runtime.close(); print(json.dumps(result, indent=2))
