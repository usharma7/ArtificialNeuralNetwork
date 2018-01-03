#initalize the neural network

from __future__ import division
import numpy as np
import csv
import random

class number_classifer():

    #sigmoid activation functioin, acts like a human nueron that supressed action until a certain threshold is reached
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))


    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #Input to Hidden Random Weights
        #create weights 100 x 784 [Hidden X Input]
        #hidden to output random weights
        # create weights  10 x 100 [output x hidden]
        input_to_hidden_weights = np.random.normal(0.0, (1/np.sqrt(self.inodes)), (self.hnodes, self.inodes))
        hidden_to_output_weights = np.random.normal(0.0, (1 / np.sqrt(self.hnodes)), (self.onodes, self.hnodes))


        #Pass needed atributes to self
        self.input_hidden_weights = input_to_hidden_weights
        self.hidden_output_weights = hidden_to_output_weights
        self.lr = learningrate
        pass



    def train(self):

        #open and buffer CSV file
        with open('/Users/utkarsh/Downloads/mnist_train.csv') as csvfile:
            file = csv.reader(csvfile, delimiter=',')
            row = list(file)


        #iterate through CSV file - currently not working due to some error in the updating of weights
        for g in range (0,10000):
            print("------------Run " + str(g) + " ------------------")
            print("***************************************************")

            # print("-----------------------input-> hidden weights ------------------")
            # print(self.input_hidden_weights)
            # print("-----------------------input-> hidden weights ------------------")


            raw_data = row[g]

            # get label and make matrix out of raw data
            number_label = raw_data[0:1]
            number_label= int(number_label[0])
            print("Expected Number: " + str(number_label))
            raw_data = raw_data[1:]
            desired_array = [int(numeric_string) for numeric_string in raw_data]
            raw_data_matrix = np.matrix(desired_array)

            # Create matrix representing expected output depending on label in first row of CSV
            expected = []
            for x in range(0, 10):
                if (x == number_label):
                    expected.append(.99)
                else:
                    expected.append(.01)

            #Normalize values of input
            input_layer = []
            for x in desired_array:
                input_layer.append((x / 255) * 0.99 + 0.01)

            #Calculate output of hidden layer
                # X = Weight_Input_to_Hidden * Inputs
                # O = sigmoid (X)
            O_hidden_layer = self.sigmoid(np.dot(self.input_hidden_weights, input_layer))

            # Calculate output of output layer
                # X = Weight_Hidden_to_Output * Hidden_Layer_Output
                # O = sigmoid (X)
            O_output_layer = self.sigmoid(np.dot(self.hidden_output_weights, O_hidden_layer))
            # print("-------Output Layer------------")
            # print(O_output_layer)
            # print("-------Output Layer------------")

            #Calculate errors
            errorOutput = expected - O_output_layer
            # print("-------Output Layer------------")
            # print(O_output_layer)
            # print("-------Output Layer------------")
            #
            # print("-------Error Output------------")
            # print(errorOutput)
            # print("----Error Output---------------")

            errorHidden = np.dot(self.hidden_output_weights.T, errorOutput)
            # print("------------error hidden---------------")
            # print(errorHidden)
            # print("------------error hidden---------------")

            # #Calculate change in the weights from Hidden -> Output Later
            # sig=  self.lr * -errorOutput
            # sh = np.reshape(sig, (10,1)) * np.reshape(self.sigmoid(O_output_layer) * (1 - self.sigmoid(O_output_layer)), (1,10))
            # delta_hidden_output = np.dot(np.reshape(sh, (1,100)), O_hidden_layer.T)
            # # print("---------------------")
            # # print(delta_hidden_output)
            # # print("---------------------")
            #
            # #update hidden -> output layer weights
            # self.hidden_output_weights += (self.hidden_output_weights - delta_hidden_output)
            #
            # print("-------Update Hidden to Output Layer--------------")
            # print(self.hidden_output_weights)
            # print("------------Update Hidden to Output Layer---------")

            #Calculate Change in weights from Input -> Hidden Layer
            #
            # ch = self.lr *errorHidden
            # change_w_input_hidden = np.reshape(ch, (10,1)) * np.reshape(self.sigmoid(O_hidden_layer) * (1 - self.sigmoid(O_hidden_layer)), (1,100))
            #
            # changeInhidden=  np.dot(change_w_input_hidden, np.matrix(input_layer).T)
            #
            # print("-------------Change In Hidden output-------------")
            # print(changeInhidden)
            # print("-------------Change In Hidden output-------------")
            #
            #
            # #update hidden -> output layer weights
            # self.hidden_output_weights = (self.hidden_output_weights - changeInhidden)
            #
            # print("-------------Weights of HIdden to Output Layer-------------")
            # print(self.hidden_output_weights)
            # print("-------------Weights of Hidden to Output Layer-------------")

            #Calculate change in input -> hidden layer weights
            deltaInput_Hidden = np.reshape(self.lr * errorOutput, (1,10) * np.reshape(self.sigmoid(O_hidden_layer) * (1- self.sigmoid(O_hidden_layer)), (100,1)))
            deltaInput_Hidden = np.reshape(deltaInput_Hidden, (self.hnodes,1))
            changeInInputtoHidden = np.dot(deltaInput_Hidden, np.reshape(np.mat(input_layer).T, (1, self.inodes)))

            #update input -> hidden layer weight s
            self.input_hidden_weights -=  (self.input_hidden_weights - changeInInputtoHidden)
            print("-------------Weights of Input to Hidden Layer-------------")
            print(self.input_hidden_weights)
            print("-------------Weights of Input to Hidden Layer-------------")


            #check the output
            for x in range (0, 10):
                max = 0
                if (O_output_layer[x]> O_output_layer[max]):
                    max = x
            self.calculatedNum= max;
            "Nueral Net Calculated Number is "
            print(self.calculatedNum)
            print()

            print("------------Run " + str(g) + " ------------------")
            print("***************************************************")
            pass


    def query(self):
        with open('/Users/utkarsh/Downloads/mnist_test.csv') as csvfile:
            file = csv.reader(csvfile, delimiter=',')
            row = list(file)

        number_correct= 0
        for x in range (0, 10000):

            current_raw_data = row[x-1]

            #get label aka actual number respresented by the raw data
            number_label= current_raw_data[0:1]
            number_label= int(number_label[0])
            print("Expected Number =" + str(number_label))
            print("-------------------")

            current_raw_data= current_raw_data[1:]
            desired_array = [int(numeric_string) for numeric_string in current_raw_data]
            raw_data_matrix = np.matrix(desired_array)

            # Normalize values of input
            input_layer = []
            for x in desired_array:
                input_layer.append((x / 255) * 0.99 + 0.01)
            print("--------Input Layer-----------")
            print(input_layer)
            print("---------Input Layer ----------")

            #Calculate Hidden Layer
                # X = W_Input_To_Hidden * Input
                # O = Sigmoid(X)
            O_hidden_layer = self.sigmoid(np.dot(self.input_hidden_weights, input_layer))

            print("------------Hidden Layer  - -----------------")
            print(O_hidden_layer.__len__())
            print("------------Hidden Layer  - -----------------")

            # Calculate Output Layer
                # X = W_Hidden_To_Output* O_Hidden_Layer
                # O = Sigmoid(X)
            O_output_layer = self.sigmoid(np.dot(self.hidden_output_weights, O_hidden_layer))

            #Check the output of the NN and compare to expected number
            max = 0
            for x in range(0, 10):
                if O_output_layer[x]> O_output_layer[max]:
                    max = x

            print("Calculated Number " + str(max))

            if max == number_label :
                number_correct +=1



        print("Percentage = " + str(number_correct/ 100000))

        pass







#for solution - if solution is 7 need to create array [0-9] with index @ 7 = .99 and all others = .01

#input layer is from CSV - google how to parse




nb = number_classifer( 784, 100, 10, .3)

nb.train()
nb.query()




