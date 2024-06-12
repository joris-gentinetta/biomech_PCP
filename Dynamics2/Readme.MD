# Dynamic Model 2nd

It is the second version of the dynamic model. In this model, there is no AMI pair anymore. The model is fully based on the muscle structure. All muscles connect to the joint at an angle; no longer that each two-muscle AMI pair controls one DoF.

## Structure

### Muscle
The file describes the dynamic of one muscle.



## PLAN

### Muscle Activation



## Training techniques

### Different Learning rates on different parts

All the parameters in the model are normalized to be one by multiplying specific factors.
The possible range decides the learning rate of a given parameter.
In this version of the implementation, the learning rates of parameters in the dynamic model are all the same.
Parameters in the neural network and the dynamic model have different learning rates.
In this way, the model can both converge quicker and more stable.

