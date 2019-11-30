import numpy as np
import random
import NNlib as nlb

class Neural:
    def __init__(self , data_matrix , batch_size , K_classes, n_hidden = 0 , n_h_neuron = 3 ):
        self.data = data_matrix
        self.n_hidden = n_hidden
        self.n_h_neuron = n_h_neuron
        self.batch_size = batch_size
        self.nbInstances = len(data_matrix)
        self.nbFeatures = len(data_matrix[0])
        self.K_classes = K_classes        
        
        self.trainingSize = int(self.nbInstances * 0.75)
        self.testingSize = self.nbInstances - self.trainingSize
        self.trainingData = np.empty(shape = (self.trainingSize , self.nbFeatures))
        self.testingData = np.empty(shape = (self.testingSize , self.nbFeatures))		
		

        self.W1 = np.empty(shape = (4 , 3) , dtype='float32')
        self.W1 = self.initMatrix(self.W1)

        self.W2 = np.empty(shape = (3 , 3) , dtype='float32')
        self.W2 = self.initMatrix(self.W2)

        

		#copy data into training and testing set
        for i in range(self.trainingSize):
            for j in range(self.nbFeatures):    
                self.trainingData[i][j] = self.data[i][j]
				
        for i in range(self.testingSize):
            for j in range(self.nbFeatures):
                self.trainingData[i][j] = self.data[i + self.trainingSize][j]
				

        self.X_train = np.empty(shape = (self.batch_size,self.nbFeatures - 1), dtype='float32')
        self.Y_train = np.empty(shape = (self.batch_size,self.K_classes) , dtype='float32')

        self.X_train = (self.X_train - np.mean(self.X_train)) / np.std(self.X_train)

        self.X_test = self.data[self.trainingSize:, :-1]
        self.Y_test = self.data[self.trainingSize:,  -1]
        self.X_test  = (self.X_test - np.mean(self.X_test)) / np.std(self.X_test)

        one_hot = np.zeros((self.Y_test.shape[0], self.K_classes))

        for i in range(self.Y_test.shape[0]):
            one_hot[i, int(self.Y_test[i])] = 1
        self.Y_test = one_hot





    def initMatrix(self , A):
        self.A = A
        for i in range(len(A)):
            for j in range(len(A[0])):  
                self.A[i][j] =  random.uniform(-.0001 , .0001)

        return self.A     


    def create_one_hot(self , k , indexInBatch , matrixY):
		# print(matrixY,"\n\n")
        for i in range(len(matrixY[0])):
            if i == k:
                matrixY[indexInBatch][i] = 1
            else:
                matrixY[indexInBatch][i] = 0
        # print(matrixY,"\n\n")
        return matrixY


    def load_attributes_labels(self , dataset , X , Y , dataindex ,batch_size):
        start_index = dataindex
        i = 0
        for row in range(start_index, start_index + batch_size):
            for col in range(self.nbFeatures - 1):
                X[i][col] = dataset[row][col]
                
            i += 1
        
        last_attribute_index = -1
        starting_index = dataindex
        for j in range(batch_size):
            self.create_one_hot(dataset[starting_index + j][last_attribute_index] , indexInBatch = j , matrixY = self.Y_train)



    def predict(self, X):
        X = np.reshape(X , (1,  4))
        H1 = np.matmul(X , self.W1)
        # print(X.shape , self.W1.shape)
        y = nlb.NNLib.sigmoid(H1)
        # y_hat = H1
        return np.reshape(y, (3, 1))


    def feed_forward(self , X):
        X = np.reshape(X , (1,  4))
        H1 = np.matmul(X , self.W1)
        # A1 = nlb.NNLib.sigmoid(H1)
        return nlb.NNLib.sigmoid(H1)
        



    def back_prop(self , y_hat , y , X):
        dL_dy = 2 * (y_hat - y)
        # dy_dH1 = dL_dy
        dy_dH1 = nlb.NNLib.sigmoid(y_hat, True).reshape(3, 1)
        dH1_dW = X

        # print(dL_dy.shape , dy_dH1.shape , dH1_dW.shape)
        dW = np.matmul(dy_dH1 * dL_dy.T , dH1_dW)

        return dW


    def error(y_hat , y ):
        y_hat = np.reshape(y_hat , (3 , 1))
        y = np.reshape(y , (3 , 1))
        error = np.mean((y_hat - y)**2)
        return error


    def train_epoch(self , n_epoch):
        self.load_attributes_labels(self.trainingData , self.X_train , self.Y_train , 0 , self.batch_size)
        epoch = n_epoch
        for j in range(epoch):
            # np.seterr(all='raise')
            total_error = 0.
            for i in range(self.batch_size):
                # ---------------   FEED FORWARD -------------
                
                X = self.X_train[i]
                X = np.reshape(X , (1,  4))

                y_hat = self.feed_forward(X)
                
                
                # --------------- BACKPROPOGATION


                y = self.Y_train[i]
                error = np.mean((y_hat - y)**2)

                total_error += error

                dW = self.back_prop(y_hat , y , X)



                # ----------UPDATE PARAMETERS -------------
                n = .001
                self.W1 -= n*dW.T



            # print("\nWeights: \n" , weg)
            # print("inputs : \n",X ,"\nH1 :\n" ,H1 , "\nyHat: \n",  y_hat , "\n lr :\n" , n)
            print(total_error / self.batch_size)
            acc = 0
            for i in range(len(self.X_test)):
                acc += nlb.NNLib.accuracy(self.predict(self.X_test[i]), self.Y_test[i])
            # print(acc / len(self.X_test) * 100)




            
            

            


        



        


        
        

