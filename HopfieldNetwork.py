import numpy as np
import random 
import matplotlib.pyplot as plt

class HopfieldNetwork:

    def __init__(self,input1,input2,initial,percent_scramble):
        """initializes BHN parameters"""
        #initiliazing the random state
        self.input1 = np.array(input1)
        self.n = self.input1.shape[1] # of neurons in network
        scramble= np.int16(np.rint(percent_scramble/100 * self.n))
        rand_idx = random.sample(range(self.n),scramble)
        # rand_idx2 = random.sample(range(784),scramble)
        # for location, idx in zip(rand_idx, np.arange(1,len(rand_idx2))): to iterate over two things in the for loop
        for location in rand_idx:
            if scramble==0:
                self.states1 = initial 
            else:
                initial[location] = -initial[location]
        self.states1 = initial
       
       #For input1 (shifted target memories)
        self.memory1 = np.array(input1) 
        self.weights1 = np.zeros((self.n,self.n)) #initializing weights matrix of zeros
        self.weights1 = (self.memory1.T@self.memory1)/self.n
     

        #For input 2 (unshifted target memories)
        self.states2 = np.copy(self.states1)
        self.input2 = np.array(input2)
        self.patterns2 = self.input2.shape[0] #number of patterns/images to be stored in memory
        self.memory2 = np.array(input2) 
        self.weights2 = np.zeros((self.n,self.n)) #initializing weights matrix of zeros
        self.weights2 = (self.memory2.T@self.memory2)/self.n
      
        



    def neuron_activation(self,number_of_neurons):
        neurons = self.input1.shape[1] 
        #For input 1 (shifted memory)
        
        self.Energy1 = [] #Energy 
        
        
        self.random_index = random.sample(range(neurons),number_of_neurons) #Choose random neuron without replacement
        self.single_state1= np.empty([number_of_neurons,neurons])
        i = 0
        for index in self.random_index:
            self.neuronactivation1 = np.dot(self.weights1[index,:],self.states1) #gives activation state of random neuron
          
            if self.neuronactivation1<0:
                self.states1[index] = -1
            else:
                self.states1[index] = 1
            self.E1 = -.5*self.states1.T @ self.weights1 @ self.states1
           
            self.single_state1[i,:] = self.states1 #To save the state of the neurons one by one
            self.Energy1.append(self.E1)
            i = i+1

            #For input 2 (unshifted memory)
                    #For input 1 (shifted memory)
        
        self.Energy2 = [] #Energy 
        
        
        self.single_state2= np.empty([number_of_neurons,neurons])
        i = 0
        for index in self.random_index:
            self.neuronactivation2 = np.dot(self.weights2[index,:],self.states2) #gives activation state of random neuron
          
            if self.neuronactivation2<0:
                self.states2[index] = -1
            else:
                self.states2[index] = 1
            self.E2 = -.5*self.states2.T @ self.weights2 @ self.states2
           
            self.single_state2[i,:] = self.states2 #To save the state of the neurons one by one
            self.Energy2.append(self.E2)
            i = i+1
            
         

        return self.states1, np.array(self.Energy1), self.single_state1, np.array(self.Energy2), self.states2