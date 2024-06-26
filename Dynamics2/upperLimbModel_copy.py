# BUGMAN Feb 1 2022
# Modified by Mikey Fernandez 06/09/2022
# 06/14/2022 - model generated for any number of DoF

"""
Upper Extremity Model parent class, can take any number of degrees of freedom
"""

import torch
from torch import nn
import numpy as np
from torch.functional import F

class upperExtremityModel(nn.Module):
    def __init__(self, device, Dynamic_Lr, EMG_mat_Lr, NN_ratio, EMG_Channel_Count=8, muscleType='bilinear', numDoF=4):
        super().__init__()

        if numDoF < 1:
            raise ValueError(f'Invalid number of degrees of freedom {numDoF}')
        
        self.device = device
        self.EMG_Channel_Count = EMG_Channel_Count
        self.EMG_mat_Lr = EMG_mat_Lr
        self.lr = Dynamic_Lr
        self.scale = 1/self.EMG_mat_Lr/self.EMG_Channel_Count

        self.numDoF = numDoF

        # make dictionary of joint model inits
        jointModels = {'bilinear': self.bilinearInit, 'shadmehr': self.shadmehrInit, 'thelen': self.thelenInit, 'crouch': self.crouchInit, 'hogan': self.hoganInit}

        # import the correct joint model
        if muscleType == 'bilinear':
            from Dynamics2.Joint_1dof_Bilinear_NN import Joint_1dof
        elif muscleType == 'shadmehr':
            from Dynamics2.Joint_1dof_Shadmehr import Joint_1dof
        elif muscleType == 'thelen':
            from Dynamics2.Joint_1dof_Thelen import Joint_1dof
        elif muscleType == 'crouch':
            from Dynamics2.Joint_1dof_Crouch import Joint_1dof
        elif muscleType == 'hogan':
            from Dynamics2.Joint_1dof_Hogan import Joint_1dof
        else:
            raise ValueError(f'Invalid muscleType {muscleType}')
        self.muscleType = muscleType

        # to ward off errors
        self.numStates = 0
        self.params = {}

        jointModels[muscleType]()
        
        self.AMIDict = []
        self.JointDict = nn.ModuleList()
        for i in range(self.numDoF):
            self.AMIDict.append([self.muscleDict[i][0], self.muscleDict[i][1]])
            self.JointDict.append(Joint_1dof(self.device, self.AMIDict[i], self.params, self.lr, NN_ratio))

        self.muscle_num = 2*self.numDoF

        # self.muscles = [self.m1, self.m2, self.m3, self.m4, self.m5, self.m6, self.m7, self.m8]
        # self.AMI1 = self.muscles[0:2]
        # self.Joint1 = Joint_1dof(self.device, self.AMI1, self.params, Dynamic_Lr, NN_ratio)

        # EMG to Muscle activation matrix
        # If the electrodes are right upon the targeted muscle, eye matrix should be chosen, otherwise all-one matrix should be the way to go
        # self.EMG_to_Activation_Mat = nn.Parameter(torch.ones((self.EMG_Channel_Count, self.muscle_num), dtype=torch.float, device=self.device)*self.scale)
        # self.EMG_to_Activation_Mat = nn.Parameter(torch.eye(self.EMG_Channel_Count, self.muscle_num , dtype=torch.float, device=self.device)*self.scale)

        # self.activationLayer = signalSeparationNetwork(device=self.device, EMG_Channel_Count=self.EMG_Channel_Count, Lr=self.scale)

        self.recognitionLayer = recognition_nn(device=self.device, numStates=self.numStates, numDoF=self.numDoF, EMG_Channel_Count=self.EMG_Channel_Count, layerSize=256)
        # self.recognitionLayer = recognition_nn(device=self.device, EMG_Channel_Count=self.EMG_Channel_Count, numDoF=self.numDoF)
        self.EMG_to_Activation_Layer = synergies_nn(device=self.device, numDoF=self.numDoF, EMG_Channel_Count=self.EMG_Channel_Count)
        """Credit assignment for separating EMG"""

    def bilinearInit(self):  
        from Dynamics2.Muscle_bilinear import Muscle

        # muscle params
        K0 = 100
        K1 = 2000
        L0 = 0.06
        L1 = 0.006
        M = 0.05

        # K0 = 1500
        # K1 = 35000
        # L0 = 0.0275
        # L1 = 0.0125
        # M = 0.0325

        self.muscleDict = []
        for _ in range(self.numDoF):
            self.muscleDict.append([Muscle(K0, K1, L0, L1, [-M]), Muscle(K0, K1, L0, L1, [M])])

        # Muscles
        # self.m1 = Muscle(100, 2000, 0.06, 0.006, [-0.05])
        # self.m2 = Muscle(100, 2000, 0.06, 0.006, [0.05])

        # self.params = {'I': [0.004], 'K': 5, 'B': .3, 'K_': 5/self.scale/self.lr, 'B_': .3/self.scale/self.lr, 'speed_mode': True, 'K0_': 2000, 'K1_': 40000, 'L0_': 0.03, 'L1_': 0.006, 'I_': 0.008, 'M_': 0.05}
        self.params = {'I': [0.004], 'K': 5, 'B': .3, 'K_': 5/self.scale/self.lr, 'B_': .3/self.scale/self.lr, 'speed_mode': False, 'K0_': 2000, 'K1_': 40000, 'L0_': 1.2, 'L1_': 0.12, 'I_': 0.064, 'M_': 0.1}
        # self.params = {'I': [0.04], 'K': 5, 'B': .3, 'K_': 80, 'B_': 4.8, 'speed_mode': True, 'K0_': 24000, 'K1_': 560000, 'L0_': 0.44, 'L1_': 0.2, 'I_': 0.64, 'M_': 0.52}

        self.numStates = 2

    def hoganInit(self):  
        from Dynamics2.Muscle_Hogan import Muscle

        # muscle params
        T = 60
        K = 10

        self.muscleDict = []
        for _ in range(self.numDoF):
            self.muscleDict.append([Muscle(T, K), Muscle(T, K)])

        self.params = {'I': [0.004], 'B': .3, 'K_': .333/self.scale/self.lr, 'B_': .01/self.scale/self.lr, 'T_': 2/self.scale/self.lr, 'I_': 0.002, 'order': 1}
        # parameters['I'], parameters['B'], Lr_scale, parameters['I_'], parameters['B_'], parameters['K_'], parameters['T_']

        self.numStates = 2
        
    def shadmehrInit(self):
        from Dynamics2.Muscle_Shadmehr import Muscle

        # lengths (muscle attachment geometry)
        # lengths = [[.1, .5], [.1, .5], [.1, .5], [.1, .5]]
        lengths = [[.1, .5]]
        L0s = np.linalg.norm(lengths, axis=1)

        # Muscle parameters: f_max, lM_opt, k_SE, k_PE, b, gamma
        fMax_def = 2000
        kSE_def = 1000
        kPE_def = kSE_def/10
        b_def = kSE_def/100
        gamma_def = 0.5

        # Joint parameters: B, K, I
        K_def = 20
        B_def = 1
        I_def = 1e-2

        self.muscleDict = []
        for _ in range(1, self.numDoF + 1):
            self.muscleDict.append([Muscle(fMax_def, L0s[0], kSE_def, kPE_def, b_def, gamma_def), Muscle(fMax_def, L0s[0], kSE_def, kPE_def, b_def, gamma_def)])
        
        # underscores denote scaling on the learning rates - learn larger parameters faster to explore the space more
        self.params = {'I': I_def, 'K': K_def, 'B': B_def, 'L':lengths, 'K_': K_def/self.scale/self.lr, 'B_': B_def/self.scale/self.lr, 'I_': I_def/self.scale/self.lr, 'fMax_': fMax_def/self.scale/self.lr, 'lOpt_': L0s[0]/self.scale/self.lr, \
           'kSE_': kSE_def/self.scale/self.lr, 'kPE_': kPE_def/self.scale/self.lr, 'b_': b_def/self.scale/self.lr, 'gamma_': gamma_def/self.scale/self.lr}

        self.numStates = 4

        # for reference
        # self.Joint1 = Joint_1dof(self.device, self.AMI1, inertias=I_def, K=K_def, B=B_def, lengths=self.lengths[0], Lr_scale=Dynamic_Lr, NN_ratio=NN_ratio, \
        #                          K_scale=K_def, B_scale=B_def, I_scale=I_def, f_max_scale=fMax_def, lM_opt_scale=L0s[0], kSEs_scale=kSE_def, kPEs_scale=kPE_def, bs_scale=b_def, gamma_scale=gamma_def)

    def thelenInit(self):
        from Dynamics2.Muscle_Thelen import Muscle

        # these initial parameters taken from https://pubmed.ncbi.nlm.nih.gov/16078622/ - approximate average upper extremity muscle
        # Muscle parameters: f_max, lM_opt, vM_max, lT_slack, pennation angle, moment arm
        fMax_def = 200 # N
        lM_opt_def = 0.06 # m
        vM_max_def = 5 # lM_opt/s
        vAlpha_def = 5 # lM_opt/s
        lT_slack_def = .25*lM_opt_def # m - I feel very subjective
        # lT_slack_def = 0.33 # this is unitless - it's given inherently in terms of ratio with lM_opt
        pen_opt_def = 10*np.pi/180 # rad
        # moment_arm_def = 0.05 # m
        lInsert_def = lM_opt_def
        lOrig_def = lM_opt_def

        # Joint parameters: B, K, I
        K_def = 2 # N-m/rad
        B_def = .1 # N-m-s/rad
        I_def = 4e-3 # kg-m^2

        # bicep-tricep
        # self.m1 = Muscle(fMax_def, lM_opt_def, vM_max_def, lT_slack_def)

        self.muscleDict = []
        for _ in range(1, self.numDoF + 1):
            self.muscleDict.append([Muscle(fMax_def, lM_opt_def, [vM_max_def, vAlpha_def], lT_slack_def, pen_opt_def, [-lInsert_def, lOrig_def]), Muscle(fMax_def, lM_opt_def, [vM_max_def, vAlpha_def], lT_slack_def, pen_opt_def, [lInsert_def, lOrig_def])])
        ## Note the way I changed the initial condition on moment arm

        # self.params = {'I': I_def, 'K': K_def, 'B': B_def, 'L':lengths, 'fMax': fMax_def, 'lOpt': lM_opt_def, 'vM': vM_max_def, 'lT': lT_slack_def}
        # self.params = {'I': [I_def], 'K': K_def, 'B': B_def, 'K_': 5/self.scale/self.lr, 'B_': .3/self.scale/self.lr, 'I_': 0.064, 'M_': 0.1, 
        #                'fMax_': fMax_def, 'lOpt_': lM_opt_def, 'vM_': vM_max_def, 'lT_': lT_slack_def, 'pen_': pen_opt_def}
        self.params = {'I': [I_def], 'K': K_def, 'B': B_def, 'K_': 5/self.scale/self.lr, 'B_': .3/self.scale/self.lr, 'I_': 0.064, 'M_': 0.1, 
                       'fMax_': fMax_def, 'lOpt_': lM_opt_def, 'vM_': vM_max_def, 'vAlpha_': vAlpha_def, 'lT_': lT_slack_def, 'pen_': pen_opt_def, 'lIn_': lInsert_def, 'lOrig_': lOrig_def}
        
        self.numStates = 4

    def crouchInit(self):
        from Dynamics2.Muscle_Crouch import Muscle

        # lengths (muscle attachment geometry)
        lengths = [[.1, .5]]
        L0s = np.linalg.norm(lengths, axis=1)

        # Muscle parameters: f_max, lM_opt, k_SE, k_PE, v_max, W, c
        fMax_def = 2000
        kPE_def = 1000
        vMax_def = 5 # changing the vMax definition!
        W_def = 0.5
        c_def = 0.25

        # Joint parameters: B, K, I
        K_def = 20
        B_def = 1
        I_def = 1e-3

        self.muscleDict = {}
        for i in range(1, self.numDoF + 1):
            keyNeg = str('m' + str(2*i - 1))
            keyPos = str('m' + str(2*i))

            self.muscleDict[keyNeg] = Muscle(fMax_def, L0s[0], kPE_def, vMax_def, W_def, c_def)
            self.muscleDict[keyPos] = Muscle(fMax_def, L0s[0], kPE_def, vMax_def, W_def, c_def)
        
        # underscores denote scaling on the learning rates - learn larger parameters faster to explore the space more
        self.params = {'I': I_def, 'K': K_def, 'B': B_def, 'L':lengths, 'K_': 320, 'B_': 16, 'I_': 16e-3, 'fMax_': 32000, 'lOpt_': 8.16, 'kPE_': 16000, 'vMax_': 160, 'W_': 8, 'c_': 4}

        self.numStates = 2

    def forward(self, SS, EMG, dt):
        # Get the muscle activations then pass them into the joint model.
        # print(f'****************************************************************************\n**************************************New timestep************************************************\n')
        # Alphas = torch.matmul(EMG[:, :self.EMG_Channel_Count], self.EMG_to_Activation_Mat*self.EMG_mat_Lr)
        Alphas = self.EMG_to_Activation_Layer(EMG)

        thetas = torch.empty((SS.shape[0], 0), device=self.device)
        states = torch.empty((SS.shape[0], 0), device=self.device)
        for i in range(self.numDoF):
            # print(f'**********************************Joint {i}**************************************\n')
            tempTh, tempStates = self.JointDict[i](SS[:, i*self.numStates:(i + 1)*self.numStates], Alphas[:, 2*i:2*(i + 1)], dt)
            thetas = torch.hstack([thetas, tempTh])
            states = torch.hstack([states, tempStates])

        predictions = self.recognitionLayer(states, Alphas, EMG[:, :self.EMG_Channel_Count])

        return thetas, states, predictions, Alphas
    
    def findInitialConditionsThelen(self, theta, omega, EMG):
        alphas = torch.matmul(EMG[:, 0:self.EMG_Channel_Count], self.EMG_to_Activation_Mat*self.EMG_mat_Lr)

        initialConditions = []
        for joint in range(self.numDoF):
            thisJoint = self.JointDict[joint]
            thisTheta = theta[:, joint]
            thisOmega = omega[:, joint]

            initialConditions.append(thisTheta)
            initialConditions.append(thisOmega)
            for muscle_num in range(thisJoint.muscle_num):
                thisAlpha = alphas[:, 2*joint + muscle_num]
                thisMuscle = thisJoint.init_muscles[muscle_num]
                lM_eq = thisMuscle.findInitlM(thisTheta, thisOmega, thisAlpha)

                initialConditions.append(lM_eq)

        initialConditions = torch.stack(initialConditions, dim=1)
        return initialConditions

    def lock_joints(self, state=True):
        for i in range(self.numDoF):
            self.JointDict[i].lock_params(state)

    def lock_EMG_mat(self, switch=True):
        self.EMG_to_Activation_Mat.requires_grad = not switch
        
    def lock_I(self, switch=True):
        for i in range(self.numDoF):
            self.JointDict[i].I.requires_grad = not switch

    def lock_for_robot(self, switch=True):
        self.lock_EMG_mat(switch)
        self.lock_I(switch)
        
    def disable_NN(self):
        for i in range(self.numDoF):
            self.JointDict[i].disable_NN()
        
    def enable_NN(self):
        for i in range(self.numDoF):
            self.JointDict[i].enable_NN()

    def set_NN(self, NN_ratio):
        for i in range(self.numDoF):
            self.JointDict[i].set_NN(NN_ratio)
        
    def print_params(self):
        for i in range(self.numDoF):
            self.JointDict[i].print_params()

        # print(f'EMG to Muscle Activation mat:\n {(self.EMG_to_Activation_Mat*self.EMG_mat_Lr).detach().cpu().numpy()}')

    def count_params(self):
        return sum(map(lambda p: p.data.numel(), self.parameters()))

class recognition_nn(nn.Module):
    """ This class is the fully connected neural network to predict which joints are being moved """
    def __init__(self, device, numStates, numDoF, EMG_Channel_Count, layerSize=256):
        """
        Args:
            
        """
        super(recognition_nn, self).__init__()
        self.device = device

        # numInputs = numStates*numDoF + 2*numDoF + EMG_Channel_Count
        numInputs = 2*numDoF + numDoF + EMG_Channel_Count
        self.numStates = numStates
        self.numDoF = numDoF

        self.net = nn.Sequential(
            nn.Linear(numInputs, layerSize),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(layerSize, layerSize),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(layerSize, layerSize),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(layerSize, numDoF),
            nn.Sigmoid()
        )

    def forward(self, stateVector, Alphas, synergies):
        # input_tensor = torch.hstack([stateVector, Alphas, synergies])
        input_tensor = torch.hstack([stateVector[:, list(range(0, 2*self.numDoF, 2))], Alphas, synergies]).to(self.device)

        output = self.net(input_tensor)

        return output

# class signalSeparationNetwork(nn.Module):
#     """ Implements a blind signal separation network according to https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=374166 """
#     def __init__(self, device, EMG_Channel_Count, Lr):
#         super(signalSeparationNetwork, self).__init__()
#         self.device = device

#         self.numEMG = EMG_Channel_Count
#         self.Lr = Lr

#         self.weights = nn.Parameter(data=torch.zeros(self.numEMG*(self.numEMG - 1), dtype=torch.float, device=self.device)) # number of non-diagonal elements
#         self.indices = torch.cat([torch.triu_indices(self.numEMG, self.numEMG, 1), torch.tril_indices(self.numEMG, self.numEMG, -1)], dim=1)

#     def forward(self, EMG):
#         weightMatrix = torch.eye(self.numEMG, dtype=torch.float, device=self.device)
#         weightMatrix[self.indices[0], self.indices[1]] = self.weights/self.Lr

#         return torch.matmul(EMG, torch.inverse(weightMatrix))

#     def print_params(self):
#         weightMatrix = torch.eye(self.numEMG, dtype=torch.float, device=self.device)
#         weightMatrix[self.indices[0], self.indices[1]] = self.weights/self.Lr

#         print(f'EMG to Muscle Activation mat:\n {weightMatrix.detach().cpu().numpy()}')

class synergies_nn(nn.Module):
    """ This class is the fully connected neural network to learn EMG coactivation"""
    def __init__(self, device, numDoF, EMG_Channel_Count):
        # Generate fully connected neural network.
        super(synergies_nn, self).__init__()

        numInputs = EMG_Channel_Count
        numOutputs = 2*numDoF
        hiddenSize = 32

        self.model = nn.Sequential(
            nn.Linear(numInputs, hiddenSize),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hiddenSize, hiddenSize),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hiddenSize, hiddenSize),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hiddenSize, numOutputs),
            nn.Sigmoid()
        )
        self.model.to(device)

    def forward(self, EMG):
        output = self.model(EMG)

        return output