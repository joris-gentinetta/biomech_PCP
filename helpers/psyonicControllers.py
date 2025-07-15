# Mikey Fernandez 07/15/2025
#
# psyonicControllers.py
# Make the controller class, running the TimeSeriesRegressorWrapper model from Joris

# import sys
# sys.path.append('/home/haptix/haptix/biomech_PCP/')

import numpy as np
import math
import torch
from helpers.models import TimeSeriesRegressorWrapper

class psyonicControllers:
	def __init__(self, arm=None, emg=None, config=None, model_path=None):
		self.emg = emg
		self.arm = arm

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = TimeSeriesRegressorWrapper(device=self.device, input_size=len(config.features), output_size=len(config.targets), **(config.to_dict()))

		self.model.load(model_path)
		self.model.to(self.device)
		self.model.eval()

		if config.model_type == 'LSTM':
			self.states = (torch.zeros(config.n_layers, 1, config.hidden_size),
					torch.zeros(config.n_layers, 1, config.hidden_size))
		else:
			self.states = self.model.model.get_starting_states(1, torch.zeros((1, 2,  len(config.targets)), device=self.device))

		#NOTE todo?
		self.targets = ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng']

		self.output_dict = {target: 0.0 for target in self.targets}

	def runModel(self):
		jointPos = self.arm.lastPosCom
		emg_timestep = np.asarray(self.emg.normedEMG)[self.emg.usedChannels]
		if not np.any(emg_timestep > 0.1): return jointPos
		emg_timestep = torch.tensor(emg_timestep, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
		with torch.no_grad():
			output, self.states = self.model.model(emg_timestep, self.states)
			output = output.squeeze().to('cpu').detach().numpy()
			for j, target in enumerate(self.targets):
				self.output_dict[target] = output[j]

				self.output_dict[target] = np.clip(self.output_dict[target], -1, 1)
				self.output_dict[target] = (self.output_dict[target] * math.pi + math.pi) / 2
			self.output_dict['thumbInPlaneAng'] = self.output_dict['thumbInPlaneAng'] - math.pi

			jointPos[0] = np.rad2deg(self.output_dict['indexAng'])
			jointPos[1] = np.rad2deg(self.output_dict['midAng'])
			jointPos[2] = jointPos[1] # np.rad2deg(self.output_dict['ringAng'])
			jointPos[3] = jointPos[1] # np.rad2deg(self.output_dict['pinkyAng'])
			jointPos[4] = np.rad2deg(self.output_dict['thumbOutPlaneAng'])
			jointPos[5] = np.rad2deg(self.output_dict['thumbInPlaneAng'])

			jointPos[0] = 1.25*(np.rad2deg(self.output_dict['indexAng'])) if jointPos[0] > 60 else jointPos[0]
			# jointPos[1] = 1.25*(np.rad2deg(self.output_dict['midAng']) - 70)
			# jointPos[2] = 1.25*(np.rad2deg(self.output_dict['ringAng']) - 70)
			# jointPos[3] = 1.25*(np.rad2deg(self.output_dict['pinkyAng']) - 70)
			jointPos[4] = 4*(np.rad2deg(self.output_dict['thumbOutPlaneAng']) - 22) # Biophys, GRU
			jointPos[5] = 4*(np.rad2deg(self.output_dict['thumbInPlaneAng']) + 20) # Biophys

		return jointPos


