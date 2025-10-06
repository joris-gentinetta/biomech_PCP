# Mikey Fernandez 01/18/2024
#
# psyonicControllers.py
# Make the controller class

# need to add the location of Junqing's model + functions to the path
import sys

# sys.path.append('/home/haptix/haptix/Upper Extremity Models/Upper Extremity Shadmehr')
# from Dynamics2.upperLimbModel import upperExtremityModel
# from Dynamics2.upperLimbModel_copy import upperExtremityModel

import torch
import numpy as np
np.set_printoptions(linewidth=200)
from helpers.BesselFilter import BesselFilterArr
import time

import argparse
import math
import os

import torch
sys.path.append('/home/haptix/haptix/biomech_PCP/')
sys.path.append('C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/helpers')
from helpers.models import TimeSeriesRegressorWrapper

import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import pandas as pd
# from time import time
import wandb
from helpers.predict_utils import Config, train_model, TSDataset, TSDataLoader
# from helpers.EMGClass import EMG
import sys
sys.path.append('/home/haptix/haptix/haptix_controller/handsim/src/')
from helpers.EMGClass import EMG
import threading

class psyonicControllers():
	def __init__(self, numMotors=6, arm=None, freq_n=3, numElectrodes=8, emg=None, config=None, model_path=None, controller_type='free_space'):
		self.emg = emg
		self.arm = arm
		self.numMotors = numMotors
		self.freq_n = freq_n
		self.numElectrodes = numElectrodes
		self.config = config
		self.controller_type = controller_type  # 'free_space' or 'interaction'

		self.probFilter = BesselFilterArr(numChannels=3, order=4, critFreqs=3, fs=self.arm.Hz, filtType='lowpass')

		self.printRate = 5 # desired Hz
		self.loopReset = int(self.arm.Hz/self.printRate)
		self.loops = 0

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# self.model = TimeSeriesRegressorWrapper(input_size=len(config.features), hidden_size=config.hidden_size,
		# 								output_size=len(config.targets), n_epochs=config.n_epochs,
		# 								seq_len=config.seq_len,
		# 								learning_rate=config.learning_rate,
		# 								warmup_steps=config.warmup_steps, num_layers=config.n_layers,
		# 								model_type=config.model_type)
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
		# self.targets = ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng', 'wristFlex']
		self.targets = ['index_Pos', 'middle_Pos', 'ring_Pos', 'pinky_Pos', 'thumbFlex_Pos', 'thumbRot_Pos']

		self.output_dict = {target: 0 for target in self.targets}

	def resetModel(self):

		# Build whole model based on muscles and masses
		self.learningRate = 5 # the dynamic system LR parameter in training __NOT__ the overall LR
		self.DoF = 4
		self.muscleType = 'bilinear'
		self.numChannels = 8
		self.system_dynamic_model = upperExtremityModel(muscleType=self.muscleType, numDoF=self.DoF, device=self.device, EMG_Channel_Count=self.numChannels, Dynamic_Lr=self.learningRate, EMG_mat_Lr=10, NN_ratio=0.2)

		# self.model_save_path = '/home/haptix/UE AMI Clinical Work/Patient Data/P7 - 453/P7_0326_2024/Trained Models/minJerkModel.tar'
		# self.model_save_path = '/home/haptix/UE AMI Clinical Work/Patient Data/P7 - 453/P7_0327_2024/Trained Models/minJerkModel_synergyNoCost.tar'
		self.model_save_path = '/home/haptix/UE AMI Clinical Work/Patient Data/C1/C1_0603_2024/Trained Models/C1_0603_2024_model.tar'
		# self.model_save_path = '/home/haptix/UE AMI Clinical Work/Patient Data/C1/C1_0603_2024/Trained Models/C1_0603_2024_CamModel.tar'

		checkpoint = torch.load(self.model_save_path, map_location=self.device)
		self.system_dynamic_model.load_state_dict(checkpoint['model_state_dict'])
		self.system_dynamic_model.to(self.device)
		self.system_dynamic_model.eval()

		# set initial conditions
		self.hidden = torch.FloatTensor([[0]*self.DoF*self.system_dynamic_model.numStates]).to(self.device)
		self.predictions = torch.zeros((1, self.DoF), dtype=torch.float, device=self.device)

		self.lastTime = time.time()

	def forwardDynamics(self):
		allEMG = self.emg.normedEMG
		usedEMG = allEMG[self.emg.usedChannels]
		EMG = torch.FloatTensor(np.array([usedEMG])).to(self.device) if not self.emg.usingSynergies else torch.FloatTensor(np.array([self.emg.synergyProd()])).to(self.device)

		with torch.no_grad():
			jointAngles, self.hidden, predictions, _ = self.system_dynamic_model(self.hidden, EMG, dt=1/self.arm.Hz)
			# self.hidden[0][0] -= 1e-10
			# self.hidden[0][1] *= 2
			# jointAngles[0] -= torch.tensor([1e-10, -1e-10, 0, 0]).to(self.device)

		# print(jointAngles[0].detach().cpu().numpy())
		# jointAngles = (jointAngles.mul(torch.tensor([1, 1, 1, 1], dtype=torch.float, device=self.device))).tanh().detach().cpu().numpy()
		jointAngles = jointAngles.detach().cpu().numpy()
		# jointAngles = (jointAngles.mul(torch.tensor([1e8, 1e7, 5e7, 1e9], dtype=torch.float, device=self.device))).tanh().detach().cpu().numpy()
		# jointAngles = ((jointAngles.mul(torch.tensor([5e8, 5e8, 5e8, 1e9], dtype=torch.float, device=self.device))).sigmoid().detach().cpu().numpy() - 0.5)*2
		probabilities = predictions.detach().cpu().numpy()[0]
		self.predictions = predictions

		# jointPos = self.arm.getCurPos()
		jointPos = self.arm.lastPosCom
		moveBool = (probabilities > np.array([0.5, 0.4, 0.995, 0.6])) if np.any(usedEMG > 0.05) else np.zeros_like(probabilities)
		# moveBool = (probabilities > np.array([0.5, 0.5, 0.2, 0.5])) if np.any(usedEMG > 0.05) else np.zeros_like(probabilities)

		# jointAngles[0] -= [0, 0, 0.7, 0]
		# print(jointAngles[0])

		# print(f'{time.time()-self.lastTime:.5f}', [f'{ang:0.3f}' for ang in jointAngles[0]], [f'{prob:0.3f}' for prob in probabilities])

		# thumbFlex = jointAngles[0][0] if jointAngles[0][0] < 0. else jointAngles[0][0] + 0.1
		thumbFlex = jointAngles[0][0] if jointAngles[0][0] < 0. else jointAngles[0][0]
		thumbRot = jointAngles[0][0]
		indAng = jointAngles[0][1]
		midAng = jointAngles[0][2]
		# rinAng = jointAngles[0][4]
		# pinAng = jointAngles[0][5]

		# print(f'{thumbFlex:06.3f} | {indAng:06.3f} | {midAng:06.3f}')

		newThumbFlex = (1 + thumbFlex)/2*120  # todo now #thumbOutPlaneAng
		newThumbRot = -(thumbRot + 1)/2*120  # thumbInPlaneAng
		newIndex = (indAng + 1)/2*120
		newMid = (midAng + 1)/2*120
		# newRing = (rinAng + 1)/2*90
		# newPinky = (pinAng + 1)/2*90

		if moveBool[0]: jointPos[4] = newThumbFlex# min(newThumbFlex, 70)
		if moveBool[0]: jointPos[5] = newThumbRot
		if moveBool[1]: jointPos[0] = newIndex 
		if moveBool[2]: jointPos[1] = newMid
		if moveBool[2]: jointPos[2] = newMid
		if moveBool[2]: jointPos[3] = newMid

		# moveBool = np.concatenate([moveBool, [True, True]], axis=0)
		# jointPos = np.where(moveBool, jointPos, self.arm.getCurPos()).tolist()

		# jointPos[5] = -66

		# if not self.loops % self.loopReset: print(f'{time.time():.5f}', [f'{ang:06.3f}' for ang in jointPos], [f'{ang:06.3f}' for ang in jointAngles[0]]); self.loops = 0

		return jointPos

# 	def runModel(self):
# 		jointPos = self.arm.lastPosCom
# 		emg_timestep = np.asarray(self.emg.normedEMG)[self.emg.usedChannels]
# 		emg_timestep = torch.tensor(emg_timestep, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
# 		with torch.no_grad():
# 			output, self.states = self.model.model(emg_timestep, self.states)
# 			output = output.squeeze().to('cpu').detach().numpy()
# 			for j, target in enumerate(self.targets):
# 				self.output_dict[target] = output[j]
# 			# self.output_dict['thumbInPlaneAng'] = self.output_dict['thumbInPlaneAng'] - math.pi
# 			# # self.output_dict['wristRot'] = (self.output_dict['wristRot'] * 2) - math.pi
# 			# self.output_dict['wristFlex'] = (self.output_dict['wristFlex'] - math.pi / 2)
# 			# # todo
# 			# # emg.printNormedEMG()
# 			#
# 			# jointPos[0] = np.rad2deg(self.output_dict['indexAng'] * math.pi)
# 			# jointPos[1] = np.rad2deg(self.output_dict['midAng'] * math.pi)
# 			# jointPos[2] = np.rad2deg(self.output_dict['ringAng'] * math.pi)
# 			# jointPos[3] = np.rad2deg(self.output_dict['pinkyAng'] * math.pi)
# 			# jointPos[4] = np.rad2deg(self.output_dict['thumbOutPlaneAng'] * math.pi)
# 			# jointPos[5] = np.rad2deg(self.output_dict['thumbInPlaneAng'] * math.pi)
# 
# 				self.output_dict[target] = self.output_dict[target].clip(-1, 1)
# 				self.output_dict[target] = (self.output_dict[target] * math.pi + math.pi) / 2
# 			# self.output_dict['wristFlex'] = self.output_dict['wristFlex'] - math.pi / 2
# 			# self.output_dict['wristRot'] = (self.output_dict['wristRot'] * 2) - math.pi
# 			# self.output_dict['thumbInPlaneAng'] = self.output_dict['thumbInPlaneAng'] - math.pi
# 
# 			# for i in range(6):
# 			# 	jointPos[i] = (np.rad2deg(self.output_dict['indexAng'] * math.pi) - 60) * 5
# 			jointPos[0] = np.min([70, np.rad2deg(self.output_dict['indexAng'])])# if np.rad2deg(self.output_dict['thumbOutPlaneAng']) > 50 else np.rad2deg(self.output_dict['indexAng'])
# 			jointPos[1] = np.rad2deg(self.output_dict['midAng'])
# 			jointPos[2] = np.rad2deg(self.output_dict['ringAng'])
# 			jointPos[3] = np.rad2deg(self.output_dict['pinkyAng'])
# 			# jointPos[4] = (np.rad2deg(self.output_dict['thumbOutPlaneAng']) - 30)*6
# 			# jointPos[5] = (np.rad2deg(self.output_dict['thumbInPlaneAng']) + 30)*6
# 			jointPos[4] = 1.5*np.rad2deg(self.output_dict['thumbOutPlaneAng'])
# 			# jointPos[5] = np.rad2deg(self.output_dict['thumbInPlaneAng'])
# 
# 			jointPos[5] = -66
# 		return jointPos

	def runModel(self, force_data=None):
		jointPos = [0]*6
		
		# Get EMG input
		emg_timestep = np.asarray(self.emg.normedEMG)[self.emg.usedChannels]
		
		if self.controller_type == 'free_space':
			model_input = emg_timestep
		elif self.controller_type == 'interaction':
			if force_data is None:
				print("Warning: Interaction controller called without force data")
				force_data = np.zeros(5)
			if len(force_data) != 5:
				print(f"Warning: Expected 5 force values, got {len(force_data)}")
				force_data = force_data[:5] if len(force_data) > 5 else np.pad(force_data, (0, 5-len(force_data)))
			# Combine EMG and force features
			model_input = np.concatenate([emg_timestep, force_data])
		else:
			raise ValueError(f"Unknown controller type: {self.controller_type}")

		# Input validation
		if len(model_input) != len(self.config.features):
			print(f"ERROR: Input size mismatch! Got {len(model_input)}, expected {len(self.config.features)}")
			print(f"Controller type: {self.controller_type}")
			print(f"EMG channels: {len(emg_timestep)}")
			if self.controller_type == 'interaction':
				print(f"Force channels: {len(force_data) if force_data is not None else 0}")
			return jointPos  # Return zeros if input size is wrong

		# DEBUG: Print EMG input
		if not hasattr(self, '_debug_counter'):
			self._debug_counter = 0
		
		self._debug_counter += 1
		# if self._debug_counter % 60 == 0:  # Every second
			# print(f"\n=== CONTROLLER DEBUG ===")
			# print(f"EMG input: {emg_timestep}")
			# print(f"EMG max: {np.max(emg_timestep):.3f}, mean: {np.mean(emg_timestep):.3f}")
		
		# Process through model
		# emg_tensor = torch.tensor(emg_timestep, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
		model_input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
		
		with torch.no_grad():
			output, self.states = self.model.model(model_input_tensor, self.states)
			output = output.squeeze().cpu().numpy()
			RESPONSIVENESS_SCALE = 2.0  # Increase this to make more responsive
			output = output * RESPONSIVENESS_SCALE
			output = np.clip(output, -1.0, 1.0)
			
			# DEBUG: Print model output
			# if self._debug_counter % 60 == 0:
			# 	print(f"Model output (raw): {output}")
			# 	print(f"Model output range: [{output.min():.3f}, {output.max():.3f}]")
			
			# Convert to joint positions
			jointPos[0] = output[0] * 60 + 60     # index_Pos
			jointPos[1] = output[1] * 60 + 60     # middle_Pos
			jointPos[2] = output[2] * 60 + 60     # ring_Pos
			jointPos[3] = output[3] * 60 + 60     # pinky_Pos
			jointPos[4] = output[4] * 60 + 60     # thumbFlex_Pos
			jointPos[5] = output[5] * 60 - 60     # thumbRot_Pos
			
			# DEBUG: Print joint positions
			# if self._debug_counter % 60 == 0:
			# 	print(f"Joint positions: {jointPos}")
			# 	print(f"Index should be at: {jointPos[0]:.1f} degrees")
		
		return jointPos
# 	def runModel(self):
# 		# Make a copy of the previous command (length 6)
# 		jointPos = [0]*6
# 
# 		# Prepare EMG input
# 		emg_timestep = np.asarray(self.emg.normedEMG)[self.emg.usedChannels]
# 		emg_timestep = torch.tensor(emg_timestep, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
# 
# 		with torch.no_grad():
# 			output, self.states = self.model.model(emg_timestep, self.states)
# 			output = output.squeeze().cpu().numpy()  # shape = (6,)
# 
# 			# Store by name (optional, but helpful)
# 			for j, target in enumerate(self.targets):
# 				self.output_dict[target] = output[j]
# 
# 			# Convert scaled outputs to raw joint angles (degrees)
# 			# First five: [0,120], last: [-120,0]
# 			jointPos[0] = output[0] * 60 + 60     # index_Pos
# 			jointPos[1] = output[1] * 60 + 60     # middle_Pos
# 			jointPos[2] = output[2] * 60 + 60     # ring_Pos
# 			jointPos[3] = output[3] * 60 + 60     # pinky_Pos
# 			jointPos[4] = output[4] * 60 + 60     # thumbFlex_Pos
# 			jointPos[5] = output[5] * 60 - 60     # thumbRot_Pos
# 
# 		return jointPos

def create_controllers(emg, arm, config_free_space, config_interaction, 
                      free_space_model_path, interaction_model_path):
    """
    Create both free space and interaction controllers.
    
    Returns:
        tuple: (free_space_controller, interaction_controller)
    """
    free_space_controller = psyonicControllers(
        numMotors=6 if arm is None else arm.numMotors,
        arm=arm,
        freq_n=3,
        emg=emg,
        config=config_free_space,
        model_path=free_space_model_path,
        controller_type='free_space'
    )
    
    interaction_controller = psyonicControllers(
        numMotors=6 if arm is None else arm.numMotors,
        arm=arm,
        freq_n=3,
        emg=emg,
        config=config_interaction,
        model_path=interaction_model_path,
        controller_type='interaction'
    )
    
    return free_space_controller, interaction_controller