import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random 
import sys
from dwave.system import DWaveSampler, EmbeddingComposite
from minorminer import find_embedding
import dwave
import dimod
import pickle
                              
def sig(x):
    #Returns sigmoid
    return 1/(1+np.exp(-x))


def sample(p):
    #Returns bernoulli sampling of array of probabilities 
    #Array of probabilities is the o/p of the sigmoid layer
    a=[]
    for pi in p:
        if np.random.rand()<pi:
            a.append(1)
        else:
            a.append(0)
    a=np.array(a)
    return a


def binarize(i,l):
    #Converts the gray scale image to a binary image
    for row in range(i.shape[0]):
        for col in range(i.shape[1]):
            if i[row][col]>=l:
                i[row][col]=1
            else:
                i[row][col]=0
    return i


def binarize_rand(i):
    #Converts the gray scale image to a binary image probabilistically
    for row in range(i.shape[0]):
        for col in range(i.shape[1]):
            if np.random.rand()<i[row][col]:
                i[row][col]=1
            else:
                i[row][col]=0
    return i

def find_img(img,train):
    '''Finds the image in train which is closest to img''' 
    L2=1e10 
    ind=0 
    for i,timg in enumerate(train): 
        if sum((timg-img)*(timg-img))<=L2: 
            ind=i 
            L2=sum((timg-img)*(timg-img)) 
    return ind 
                 

class model_save(): 
    '''Dummy class for saving parameter values after training'''
    def __init__(self,N,M,L,J,h,Wg,bwg,Vg,bvg,Wr,bwr,Vr,bvr): 
        self.N=N 
        self.M=M 
        self.L = L 
        self.J = J 
        self.h = h 
        self.Wg = Wg 
        self.bwg = bwg 
        self.Vg = Vg 
        self.bvg = bvg                                                                                                                


class model(): 
    '''Class containing methods to create the networks and train them'''
    def __init__(self,N,M,L,annealing_time,programming_thermalization,readout_thermalization):
        self.N = N
        self.M = M
        self.L = L
        self.Vr = np.zeros((M,N))
        self.Wr = np.zeros((L,M))
        self.Wg = np.zeros((M,L))
        self.Vg = np.zeros((N,M))
        self.J = np.zeros((L,L))
        self.h = np.zeros(L)
        self.bvr = np.zeros((M,))
        self.bwr = np.zeros((L,))
        self.bwg = np.zeros((M,))
        self.bvg = np.zeros((N,))
        self.annealing_time=annealing_time
        self.programming_thermalization=programming_thermalization
        self.readout_thermalization=readout_thermalization
        
        self.hidden_graph=[] #Adjacency Matrix of the hidden layer
        for i in range(L):
            #Creates the abovesaid Adjacency Matrix
            for j in range(i+1,L):
                self.hidden_graph.append((i,j))

        self.sampler_manual=DWaveSampler(solver={'qpu':True}) #DWave sampler instance
        self.embedding=find_embedding(self.hidden_graph,self.sampler_manual.edgelist,random_seed=10)   #Calculates and stores heuristic embedding
        


    def qsample(self,J,h,n_read):
    #Returns samples from the quantum device

        self.hdict={}
        self.Jdict={}
        # print("Quantum Sampling")
        for i,hi in enumerate(h):
            self.hdict[i]=hi                #Creates the bias dictionary   
        for i in range(L):
            for j in range(i+1,L):
                self.Jdict[(i,j)]=J[i][j]   #Creates the interaction dictionary from matrix

        #Embed        
        self.th, self.tJ = dwave.embedding.embed_ising(self.hdict,self.Jdict,self.embedding,self.sampler_manual.adjacency) #Embeds the particular biases/interactions
        #Sample
        self.sampleset=self.sampler_manual.sample_ising(self.th,self.tJ,num_reads=n_read,answer_mode='raw',annealing_time=self.annealing_time,programming_thermalization=self.programming_thermalization,readout_thermalization=self.readout_thermalization) 
        #Unembed the data
        self.bqm=dimod.BinaryQuadraticModel.from_ising(self.hdict,self.Jdict)
        self.samples=dwave.embedding.unembed_sampleset(self.sampleset,self.embedding,self.bqm)
        
        #Convert data into bits from spins and return
        return (self.samples.record.sample+1)/2


        
    def wake(self,data,lr):
        #Does the wake phase gradient update
        #Samples the recognition network
        self.d = data
        self.y = sample(sig(np.matmul(self.Vr,self.d)+self.bvr))
        self.x = sample(sig(np.matmul(self.Wr,self.y)+self.bwr))
                
        #Passes back down through the generative network to find relevant probabilities
        self.psi = sig(np.matmul(self.Wg,self.x)+self.bwg)
        self.delta = sig(np.matmul(self.Vg,self.y)+self.bvg)

        #Computing gradients and updating parameters as mentioned in the report 
        self.J = self.J - lr*np.outer(self.x,self.x)
        self.h = self.h - lr*self.x
        
        self.Wg = self.Wg+lr*np.outer((self.y-self.psi),self.x)
        self.bwg = self.bwg+lr*(self.y-self.psi)
        self.Vg = self.Vg+lr*np.outer((self.d-self.delta),self.y)
        self.bvg = self.bvg+lr*(self.d-self.delta)
        
    def sleep(self,lr):
        #Does the sleep phase gradient update
        
        #Samples the generative network
        self.x = self.qsample(self.J,self.h,1)[0]
        self.y = sample(sig(np.matmul(self.Wg,self.x)+self.bwg))
        self.d = sig(np.matmul(self.Vg,self.y)+self.bvg)  

        #Passes back through recognition network to compute relevant probabilities
        self.psi = sig(np.matmul(self.Vr,self.d)+self.bvr)
        self.eta = sig(np.matmul(self.Wr,self.y)+self.bwr)
        self.Vr = self.Vr+lr*np.outer((self.y-self.psi),self.d)
        self.bvr = self.bvr+lr*(self.y-self.psi)
        self.Wr = self.Wr+lr*np.outer((self.x-self.eta),self.y)
        self.bwr = self.bwr+lr*(self.x-self.eta)
        

    def samplegen(self,N):
        #Returns N samples of the generative network
        l=[]
        self.qsamples=self.qsample(self.J,self.h,N)
        for i in range(N):
            self.x = self.qsamples[i]
            self.y = sample(sig(np.matmul(self.Wg,self.x)+self.bwg))
            self.d = sig(np.matmul(self.Vg,self.y)+self.bvg) 
            l.append(self.d)
        return l

    def print_params(self):
        #Prints the values of all the parameters
        print("N = ", self.N)
        print("M = ", self.M)
        print("N = ", self.L)
        
        print("Vr = ", self.Vr)
        print("bvr = ", self.bvr)
        
        print("Wr = ", self.Wr)
        print("bwr = ", self.bwr)
        
        print("Wg = ", self.Wg)
        print("bwg = ", self.bwg)
        
        print("Vg = ", self.Vg)
        print("bvg = ", self.bvg)
        
        print("J = ", self.J)
        print("h = ", self.h)

    def save_params(self):
        '''Saves parameters in a pickle file'''
        Model_save=model_save(Model.N,Model.M,Model.L,Model.J,Model.h,Model.Wg,Model.bwg,Model.Vg,Model.bvg,Model.Wr,Model.bwr,Model.Vr,Model.bvr) 
        with open('model_save_final_final.pickle','wb') as f:
            pickle.dump(Model_save,f)
    
    def load_params(self):
        '''Loads parameters from a pickle file'''
        with open('model_save_final.pickle', 'rb') as f: 
            Model_load = pickle.load(f)
        self.N = Model_load.N 
        self.M = Model_load.M 
        self.L = Model_load.L 
        self.Vr = Model_load.Vr 
        self.Wr = Model_load.Wr 
        self.Wg = Model_load.Wg 
        self.Vg = Model_load.Vg 
        self.J = Model_load.J 
        self.h = Model_load.h 
        self.bvr = Model_load.bvr                                                                                                        
        self.bwr = Model_load.bwr 
        self.bwg = Model_load.bwg 
        self.bvg = Model_load.bvg 

#Quantum hyperparameters in microseconds
annealing_time=5
programming_thermalization=1
readout_thermalization=1

#Network dimensions
N=784
M=120
L=50

#Creating the model
Model=model(N,M,L,annealing_time,programming_thermalization,readout_thermalization)


#Loading the dataset
print("Loading dataset")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Normalizing the data
x_train=x_train/256.0
x_test=x_test/256.0
train=[]

print("Creating image list")
for i,img in enumerate(x_train):
    train.append(np.reshape(img,784)) 
for i,img in enumerate(x_test):
    train.append(np.reshape(img,784))


#Shuffling the data    
random.shuffle(train)

#Other hyperparameters
n_epoch=50
mb_size=60
#Creates the learning rate variation
lr=np.concatenate((np.ones(int(n_epoch/2.0))*0.01,np.linspace(0.01,0.001,n_epoch-int(n_epoch/2.0))))


#Model training
for i in range(n_epoch):
    rand_index=np.random.randint(0,70000,mb_size) #Choses random images
    for j,index in enumerate(rand_index):
        #Model is trained, image by image
        Model.wake(train[index],lr[i])
        Model.sleep(lr[i])
    print("Epoch number: ",i)


#Saving parameters
Model.save_params()

#Generating samples
n_samples=20
l=Model.samplegen(n_samples)

for i in range(len(l)):
    plt.imshow(np.reshape(l[i],(28,28)),cmap="gray")
    plt.title("Generated Image")
    plt.savefig('Generated image_{}.png'.format(i))
    plt.show()
    #Finding the image in the training set which is closest by L2 norm
    ind=find_img(l[i],train) 
    plt.imshow(np.reshape(train[ind],(28,28)),cmap='gray') 
    plt.title("Closest Image {}".format(i)) 
    plt.savefig("Closest Image {}".format(i)) 
    plt.show() 
