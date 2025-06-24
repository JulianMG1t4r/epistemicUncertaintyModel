import sys
import time
import multiprocessing as mp
import math
import scipy
import scipy.optimize
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import r2_score

start_time = time.time()
multiprocessing = True

class EpistemicUncertaintyModel():
    def __init__(self, input_dim, feature_dim, alphaExponent, sigma=1.0, beta=0.0,threshhold1= 10,threshhold2=10):
        #print("Creating Model..")
        #print("--- %s seconds ---" % (time.time() - start_time))

        #Input Parameter
        self.input_dim = input_dim
        if doRegression:
            self.input_dim = input_dim + 1
        self.feature_dim = feature_dim
        self.sigma = sigma
        self.alpha = 1*pow(10,alphaExponent)
        self.beta = 1*pow(10,beta)
        self.threshholdfactor = threshhold1
        self.threshholdfactor2 = threshhold2
        self.slope = 1.0
        #Variables 
        self.p = 0
        self.lernrate = 0.1
        self.deltaTime = 0
        #Compute 
        self.W_in = self.Winp()
        self.b = self.bias()
        if doRegression:
            self.features = self.computeFeatures(trainingSet)
            self.Sn = self.computeSn(self.features)
            self.threshhold = 0
        if doVolume:
            self.features = self.computeFeatures(trainingSet)
            self.Sn = self.computeSn(self.features)
            self.threshhold = self.computeThreshhold(self.Sn,self.features,self.threshholdfactor)
        if doClassificationWithInDistributionDetection:
            #Class1
            self.features1 = self.computeFeatures(trainingClass1)
            self.Sn1 = self.computeSn(self.features1)
            self.threshhold1 = self.computeThreshhold(self.Sn1,self.features1 ,self.threshholdfactor)
            #Class2
            self.features2 = self.computeFeatures(trainingClass2)
            self.Sn2 = self.computeSn(self.features2)
            self.threshhold2 = self.computeThreshhold(self.Sn2,self.features2 ,self.threshholdfactor2)
        if getTime:
            self.features = self.computeFeatures(trainingSet)
            time1 = (time.time() - start_time)
            self.Sn = self.computeSn(self.features)
            time2 = (time.time() - start_time)
            self.deltaTime = time2 - time1
        if showColorGradient:
            self.features = self.computeFeatures(trainingSet)
            self.Sn = self.computeSn(self.features)
            self.threshhold = self.computeThreshhold(self.Sn,self.features,self.threshholdfactor)

        #print("Model creation complete")
        #print("--- %s seconds ---" % (time.time() - start_time))

    def getDeltaTime(self):
        return self.deltaTime

    def activation(self,x):
        r= 1/(1+np.exp(-self.slope*x)) # logistic
        #r = x/(1+abs(x))
        #r = np.maximum(0,x) # ReLU
        #r = np.tanh(x)
        return r

    def Winp(self):
        A = np.random.randn(self.feature_dim, self.input_dim)
        #A = np.array([])
        #for j in range(0,self.feature_dim):
        #    B = np.array([])
        #    for i in range(0,self.input_dim):
        #        sample = np.random.normal(0,self.sigma) # Normalverteilung
        #        #sample = np.random.uniform(-self.sigma,self.sigma) # uniforme Verteilung
        #        #sample = np.random.choice([-self.sigma/10,self.sigma/10],1)[0] # diskrete Verteilung
        #        B = np.append(B,sample)   
        #    A = np.append(A,B)
        #A = A.reshape(self.feature_dim,self.input_dim)
        return A

    def bias(self):
        b = np.random.randn(self.feature_dim)
        #b = np.array([])
        #for i in range(0,self.feature_dim):
        #    b = np.append(b,np.random.normal(0,self.sigma))
            #b =  np.append(b,np.random.uniform(-self.sigma,self.sigma))
            #b = np.random.choice([-self.sigma/10,self.sigma/10],1)[0]
        return b

    def phi(self,x): #point in input space
        x = np.array(x)
        dot = np.dot(x,self.W_in.transpose())
        return self.activation(dot +self.b)#

    def epistemic(self,p,Sn): # Unsicherheit an Punkt p für p aus dem Merkmalsraum
        dot = np.dot(p.transpose(),Sn)
        return np.dot(dot,p)
    
    def epistemicSn(self,x): #point in feature spaces
        p = self.phi(np.append(self.p,np.array([x])))
        dot = np.dot(p.transpose(),self.Sn)
        return np.dot(dot,p)

    def gradient(self,x,Sn): #point in input space
        p = self.phi(x)
        t1 = np.dot(p,Sn)
        t2 = np.dot(p,(1 - p))
        gradient = np.dot(t1, 2*self.W_in) * t2
        return gradient
    
    def gradientSn(self,x): #point in input space
        p = self.phi(np.append(self.p,np.array([x])))
        t1 = np.dot(p,self.Sn)
        t2 = np.dot(p,(1 - p))
        gradient = np.dot(t1, 2*self.W_in) * t2
        return gradient[-1]

    def pointInRect(self,point,rect):
        x1, y1, w, h = rect
        x2, y2 = x1+w, y1+h
        x, y = point
        if (x1 < x and x < x2):
            if (y1 < y and y < y2):
                return True
        return False

    def computeThreshhold(self,Sn,set,factor):
        #print("Compute Threshhold: ")
        #print("--- %s seconds ---" % (time.time() - start_time))
        xMin = math.inf
        xMax = 0
        for data in set:
            e = self.epistemic(data,Sn)
            if e < xMin:
                xMin = e
            if e > xMax:
                xMax = e
        return factor*xMin
    
    def computeFeatures(self,trainingSet):
        #print("Compute Features:")
        #print("--- %s seconds ---" % (time.time() - start_time))
        #if self.feature_dim > 100:
        if multiprocessing:
            c = len(trainingSet)//mp.cpu_count()+10
            pool = mp.Pool(mp.cpu_count())
            f = pool.map(self.phi,trainingSet, chunksize=c)
            features = np.array([f])
            features = features.reshape(len(trainingSet),self.feature_dim)
        else:
            features = np.array([])
            for t in trainingSet:
                features = np.append(features,self.phi(t))
            features = np.array(features)
            features = features.reshape(len(trainingSet),self.feature_dim)
        return features
    
    def computeSn(self,features): # Präzisionsmatrix Sn
        Sn_inv = self.alpha * np.identity(self.feature_dim) + self.beta * np.dot(features.transpose(),features)
        Sn = np.linalg.inv(np.array(Sn_inv))
        return Sn

    def evaluateModel(self,useTestset, showColorGradient=False):
        if doClassificationWithInDistributionDetection:
            #self.ClassificationWithMinimumUncertainty(self.Sn1,self.Sn2,self.threshhold1,self.threshhold2,useTestset,True) # Train Model
            acc = self.ClassificationWithMinimumUncertainty(self.Sn1,self.Sn2,self.threshhold1,self.threshhold2,useTestset) # Evaluate Model

        if doRegression:
            acc = self.regression(useTestset)
        
        if showColorGradient:
            acc = self.colorGradient(self.Sn)

        if doVolume:
            acc = self.volume()

        return acc
    
    def volume(self):
        print("Volume")
        inVolumeTrue = 0
        outVolumeFalse = 0
        inVolume = 0

        resolutionB = 100
        resolutionX = 100

        epistemicX = np.array([])
        epistemicY = np.array([])

        epistemicZB = np.array([])

        epistemicX1 = np.array([])
        epistemicY1 = np.array([])
        epistemicZ1 = np.array([])
        epistemicX2 = np.array([])
        epistemicY2 = np.array([])
        epistemicZ2 = np.array([])

        bi = -bildausschnitt/2
        bX = (bildausschnitt/resolutionX)
        bY = (bildausschnitt/resolutionB)

        for i in range(0,resolutionX):
            #print(i , "/" , resolutionX)
            for j in range(0,resolutionB):
                x = bi + bX * i
                y = bi + bY * j
                if not useDataSet:
                    z = self.phi([x,y])
                    e = self.epistemic(z,self.Sn)
                    
                    if doVolume:
                        if (e < self.threshhold) and self.pointInRect(tuple([x,y]),rect):
                            epistemicX1 = np.append(epistemicX1,x)
                            epistemicY1 = np.append(epistemicY1,y)
                            epistemicZ1 = np.append(epistemicZ1,e)
                            inVolumeTrue = inVolumeTrue + 1
                        elif (e < self.threshhold) and not self.pointInRect(tuple([x,y]),rect):
                            epistemicX = np.append(epistemicX,x)
                            epistemicY = np.append(epistemicY,y)
                            epistemicZB = np.append(epistemicZB,e)
                            outVolumeFalse = outVolumeFalse + 1
                        else:
                            epistemicX2 = np.append(epistemicX2,x)
                            epistemicY2 = np.append(epistemicY2,y)
                            epistemicZ2 = np.append(epistemicZ2,e)
                        
                        if self.pointInRect(tuple([x,y]),rect):
                            inVolume = inVolume + 1 
                    else:
                        epistemicX = np.append(epistemicX,x)
                        epistemicY = np.append(epistemicY,y)
                        epistemicZB = np.append(epistemicZB,e)

        if doVolume:
            if inVolume > 0:
                acc = (inVolumeTrue - outVolumeFalse) / inVolume
            else:
                acc = 0
            if acc < 0:
                acc = 0
        print(acc)

        # e = Unsicherheit am Datenpunkt
        # [x,y] = Datenpunkt
        if (e < self.threshhold) and self.pointInRect(tuple([x,y]),rect): # Punkte die richtig als in der Verteilung liegend klassifiziert wurden
            inVolumeTrue = inVolumeTrue + 1
        elif (e < self.threshhold) and not self.pointInRect(tuple([x,y]),rect): # Punkte die außerhalb der Verteilung liegen und falsch klassifiziert wurden
            outVolumeFalse = outVolumeFalse + 1
        if self.pointInRect(tuple([x,y]),rect): # Punkte die in der Verteilung liegen
            inVolume = inVolume + 1 

        return acc
        
    def evaluateModelNtimes(self,n, useTestset, showColorGradient=False):
        print("Evaluate Model n times")
        print("--- %s seconds ---" % (time.time() - start_time))
        testAcc = np.array([])
        for i in range(n):
            self.W_in = self.Winp()
            self.b = self.bias()
            if doRegression:
                self.features = self.computeFeatures(trainingSet)
                self.Sn = self.computeSn(self.features)
                self.threshhold = 0
            if doVolume:
                self.features = self.computeFeatures(trainingSet)
                self.Sn = self.computeSn(self.features)
                self.threshhold = self.computeThreshhold(self.Sn,self.features,self.threshholdfactor)
            if doClassificationWithInDistributionDetection:
                #Class1
                self.features1 = self.computeFeatures(trainingClass1)
                self.Sn1 = self.computeSn(self.features1)
                self.threshhold1 = self.computeThreshhold(self.Sn1,self.features1 ,self.threshholdfactor)
                #Class2
                self.features2 = self.computeFeatures(trainingClass2)
                self.Sn2 = self.computeSn(self.features2)
                self.threshhold2 = self.computeThreshhold(self.Sn2,self.features2 ,self.threshholdfactor)
    
            acc = self.evaluateModel(useTestset, showColorGradient=showColorGradient)
            testAcc = np.append(testAcc,acc)
        print(testAcc)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("End: Evaluate Model n times")
        return testAcc

    def ClassificationWithMinimumUncertainty(self,SN1,SN2,Threshhold1,Threshhold2, useTestset=False, train=False):
        #print("ClassificationWithMinimumUncertainty:")
        #print("--- %s seconds ---" % (time.time() - start_time))
        #advancedClassifi1 = True
        ClassifiMinimumUncertainty = False
        ClassifiInDistribution = False

        #print(Threshhold1)
        #print(Threshhold2)

        correctlyClassified = 0
        wronglyClassified = 0
        case1 = 0
        case2 = 0
        case3 = 0
        case4 = 0
        class1 = 0
        correctClass1 = 0
        correctClass2 = 0

        averageUncertainty1 = 0
        averageUncertainty2 = 0
        averageUncertainty3 = 0
        averageUncertainty4 = 0

        if train:
            numberOfPointsToClassifi = trainingset_size
            Set = trainingSet
            Labels = trainingLabels
        else:
            if useTestset:
                numberOfPointsToClassifi = testset_size
                Set = testSetScaled
                Labels = testLabels
            else: 
                numberOfPointsToClassifi = validationset_size
                Set = validationSet
                Labels = validationLabels

        for i in range(numberOfPointsToClassifi):
            #print(i)
            #r = np.random.randint(0,validationSetSize)
            #r = i
            p1 = Set[i]
            #print(p)
            p = self.phi(p1)
            u1 = self.epistemic(p,SN1)
            #averageUncertainty1 = averageUncertainty1 + epistemic1
            #NormalizeUncertainty(epistemic1,Threshhold1[0],Threshhold1[1])
            #averageUncertainty1 = averageUncertainty1 + epistemic1
            u2 = self.epistemic(p,SN2)
            #h3 = self.epistemic(p,self.Sn)
            #NormalizeUncertainty(epistemic2,Threshhold2[0],Threshhold2[1])
            #averageUncertainty2 = averageUncertainty2 + epistemic2
            #print(trainingSet[nearestPointIndex])
            #print(validationLabels[i])
            #print("epistemic1 " + str(epistemic1))
            #print("epistemic2 " + str(epistemic2))
            factor = 1000
            if ClassifiMinimumUncertainty:
                u1 = 1-np.tanh(u1*Threshhold1*factor)#*self.activation(Threshhold1)
                u2 = 1-np.tanh(u2*Threshhold2*factor)#*self.activation(Threshhold2) 
                #h3 = 1-np.tanh(h3*self.threshhold*factor)
                #print('epistemic1:',h1)
                #print('epistemic2:',h2)
                out1 = (u1 * self.w1 + u2 * self.w2)
                out2 = (u2 * self.w4 + u1 * self.w3)
                o = scipy.special.softmax(np.array([out1,out2]))
                out1 = o[0]
                out2 = o[1]
                #print(out1,out2)
                if train:
                    t = np.array([1-Labels[i],Labels[i]])
                    deltao1 = -(t[0]-o[0])*o[0]*(1-o[0])
                    deltao2 = -(t[1]-o[1])*o[1]*(1-o[1])
                    self.w1 = self.w1 - self.lernrate * deltao1 * u1
                    self.w2 = self.w2 - self.lernrate * deltao1 * u2
                    self.w3 = self.w3 - self.lernrate * deltao2 * u1
                    self.w4 = self.w4 - self.lernrate * deltao2 * u2
                    #self.w5 = self.w5 - self.lernrate * deltao1 * h3
                    #self.w6 = self.w6 - self.lernrate * deltao2 * h3
                    #print(self.w1,self.w2,self.w3,self.w4,self.w5,self.w6)
                if out1 > out2:
                    label = class1Label
                    averageUncertainty1 = averageUncertainty1 + u1
                    averageUncertainty2 = averageUncertainty2 + u2
                    case1 = case1 + 1
                    case3 = case3 + 1
                else:
                    label = class2Label
                    averageUncertainty3 = averageUncertainty3 + u1
                    averageUncertainty4 = averageUncertainty4 + u2
                    case4 = case4 + 1
                    case2 = case2 + 1
            elif ClassifiInDistribution:
                if u1 < Threshhold1 and (not u2 < Threshhold2): # Fall 1
                    label = class1Label
                    averageUncertainty1 = averageUncertainty1 + u1
                    averageUncertainty2 = averageUncertainty2 + u2
                    case1 = case1 + 1
                elif u2 < Threshhold2 and (not u1 < Threshhold1): # Fall 2
                    label = class2Label
                    averageUncertainty3 = averageUncertainty3 + u1
                    averageUncertainty4 = averageUncertainty4 + u2
                    case2 = case2 + 1
                elif u1 < u2: # Fall 3
                    label = class1Label
                    averageUncertainty1 = averageUncertainty1 + u1
                    averageUncertainty2 = averageUncertainty2 + u2
                    case3 = case3 + 1
                else: # Fall 4
                    label = class2Label
                    averageUncertainty3 = averageUncertainty3 + u1
                    averageUncertainty4 = averageUncertainty4 + u2
                    case4 = case4 + 1
            else:
                #if (epistemic1 - AverageUncertainty1) < (epistemic2 - AverageUncertainty2):
                #if (epistemic1 / AverageUncertainty1) < (epistemic2 / AverageUncertainty2):
                if u1 < u2: # and  epistemic2 > AverageUncertainty2:
                #if epistemic1 < 0.4:#(averageUncertainty1/i):
                    label = class1Label
                    averageUncertainty1 = averageUncertainty1 + u1
                    averageUncertainty2 = averageUncertainty2 + u2
                    case1 = case1 + 1
                else:
                    label = class2Label
                    averageUncertainty3 = averageUncertainty3 + u1
                    averageUncertainty4 = averageUncertainty4 + u2 
                    case2 = case2 + 1

            if Labels[i] == label:
                correctlyClassified += 1
                if label == class1Label:
                    correctClass1 = correctClass1 + 1
                else:
                    correctClass2 = correctClass2 + 1
            else:
                wronglyClassified += 1

            if Labels[i] == class1Label:
                class1 = class1 + 1
            
        #error = wronglyClassified / (wronglyClassified + correctlyClassified)
        accuracy = correctlyClassified / (numberOfPointsToClassifi)

        #print("Classificaton Error:", error)
        
        #print("PercentageOfClass1: ", class1 / (wronglyClassified + correctlyClassified))
        #print("Case1: ", case1 / (numberOfPointsToClassifi))
        #print("Case2: ", case2 / (numberOfPointsToClassifi))
        #print("Case3: ", case3 / (numberOfPointsToClassifi))
        #print("Case4: ", case4 / (numberOfPointsToClassifi))
        #print(w11)
        #print(w21)
        #print(w22)
        #print(w12)
        #print("AccuracyClass1: ", correctClass1 / (class1))
        #print("AccuracyClass2: ", correctClass2 / (numberOfPointsToClassifi-class1))
        #print("AverageUncertaintyClass1: ", averageUncertainty1 / (numberOfPointsToClassifi))
        #print("AverageUncertaintyClass2: ", averageUncertainty2 / (numberOfPointsToClassifi))
        #if (case1 + case3) > 0:
        #    print("AverageUncertaintyCase13Space1: ", averageUncertainty1 / (case1 + case3))
        #    print("AverageUncertaintyCase13Space2: ", averageUncertainty2 / (case1 + case3))
        #if (case2 + case4) > 0:
        #    print("AverageUncertaintyCase24Space1: ", averageUncertainty3 / (case2 + case4))
        #    print("AverageUncertaintyCase24Space2: ", averageUncertainty4 / (case2 + case4))
        if train:
            print("train")
            print(self.w1,self.w2,self.w3,self.w4,self.w5,self.w6)
        #print("Accuracy:", accuracy)
        return accuracy

    def regression(self,useTestset):
        predictedValues = np.array([])
        if useTestset:
            trueValues = testLabels
            for i in range(testset_size):
                self.p = testSetScaled[i]
                prediction = self.minimize(0)
                predictedValues = np.append(predictedValues,prediction)
        else: 
            trueValues = validationLabels
            for i in range(validationset_size):
                self.p = validationSet[i]
                prediction = self.minimize(0)
                predictedValues = np.append(predictedValues,prediction)
                
        acc = r2_score(trueValues, predictedValues)
        #print(trueValues)
        #print(predictedValues)
        return acc
    
    def minimize(self,start):
        res = scipy.optimize.minimize(self.epistemicSn, start, method='BFGS', jac=self.gradientSn, options={'disp': False, 'maxiter':100, 'gtol':0.001})
        return res.x

    def nextFigure(self):
        global figure
        figure = figure + 1
        plt.figure(figure)

    def plotGradient(self,start,Sn): #point in input space
        global lernrate
        startPunkt = np.array(start)
        gradientX = np.array([])
        gradientY = np.array([]) 
        gradientX = np.append(gradientX,startPunkt[1])
        gradientY = np.append(gradientY,startPunkt[0])

        for i in range(0,100):
            g = self.gradient(np.array(startPunkt),Sn)
            point = startPunkt - lernrate * g 
            #point = startPunkt - lernrate * g/np.linalg.norm(g) #* -np.log10(epistemic(phi(startPunkt)))
            gradientX = np.append(gradientX,point[1])
            gradientY = np.append(gradientY,point[0])
            startPunkt = point
            #if epistemic(phi(point),Sn) < threshhold:
            #    break
        plt.plot(gradientX,gradientY,'r',linewidth=1)

        return startPunkt

    def plotEpistemicUncertaintyAndDataPoints(self,b,winp,SnD,x,y,z,x1,y1,z1,x2,y2,z2,x3,y3,z3,dotSize, dataClass1X, dataClass1Y,dataClass2X,dataClass2Y, dimention,acc):
        self.nextFigure()
        n = colors.LogNorm()
        title = "Accuracy:" + str(acc) + "Alpha: " + str(alphaExponentTemp) + " Sigma: " + str(sigmaTemp) + " Beta: " + str(betaTemp) + "Dimention: " + str(featureDimentionTemp) + "threshholdfactor: " + str(threshholdfactor1Temp)
        plt.title(title)
        plt.scatter(x1,y1,dotSize,z1,cmap = "Reds", norm= n)
        plt.scatter(x2,y2,dotSize,z2,cmap = "Greens", norm= n)
        plt.scatter(x,y,dotSize,z,cmap = "Purples", norm= n)
        #plt.scatter(x3,y3,dotSize,z3,cmap = "Oranges", norm= n)
        plt.scatter(x3,y3,dotSize,z3,cmap = "YlGnBu", norm= n)
        plt.plot(dataClass1X,dataClass1Y, 'go', markersize=0.5)
        plt.plot(dataClass2X,dataClass2Y, 'yo', markersize=0.5)
        plt.axis([-bildausschnitt/2,bildausschnitt/2,-bildausschnitt/2,bildausschnitt/2])
        #plt.axis([0,1,0,1])

        if showVectorField:
            self.vectorField(b,winp,SnD,dimention)
    
    def colorGradient(self,Sn):
        print("EpistemicUcertaityArea")

        if True:
            resolutionB = 100
            resolutionX = 100
            dotSize = 10

            epistemicX = np.array([])
            epistemicY = np.array([])
            epistemicZB = np.array([])

            epistemicX1 = np.array([])
            epistemicY1 = np.array([])
            epistemicZ1 = np.array([])
            epistemicX2 = np.array([])
            epistemicY2 = np.array([])
            epistemicZ2 = np.array([])

        bi = -bildausschnitt/2
        bX = (bildausschnitt/resolutionX)
        bY = (bildausschnitt/resolutionB)

        for i in range(0,resolutionX):
            #print(i , "/" , resolutionX)
            for j in range(0,resolutionB):
                x = bi + bX * i
                y = bi + bY * j
                if not useDataSet:
                    z = self.phi([x,y])
                    e = self.epistemic(z,Sn)
                    
                if not useDataSet:
                    epistemicX = np.append(epistemicX,x)
                    epistemicY = np.append(epistemicY,y)
                    epistemicZB = np.append(epistemicZB,e)

        if not useDataSet:
            self.nextFigure()
            n = colors.LogNorm()
            plt.scatter(epistemicX1,epistemicY1,dotSize,epistemicZ1,cmap = "Reds")
            plt.scatter(epistemicX2,epistemicY2,dotSize,epistemicZ2,cmap = "Greens")
            plt.scatter(epistemicX,epistemicY,dotSize,epistemicZB,cmap = "YlGnBu", norm= n) 
            #Plot Data Points
            inputX = np.array([])
            inputY = np.array([])
            trainingset_size = 1000
            for i in range(0,trainingset_size):
                inputX = np.append(inputX,trainingSet[i,0])
                inputY = np.append(inputY,trainingSet[i,1])
            
            plt.plot(inputX,inputY, 'go', markersize=1)

        if showVectorField:
            self.vectorField(Sn)
        
        return 0

    def vectorField(self,Sn):
        #print("VectorField")
        #Plot VectorField
        plt.axis([-bildausschnitt/2,bildausschnitt/2,-bildausschnitt/2,bildausschnitt/2])
        resolutionY = 20
        resolutionX = 20
        vectorX = np.array([])
        vectorY = np.array([])
        vectorU = np.array([])
        vectorV = np.array([])
        bi = -bildausschnitt/2
        bx = (bildausschnitt/resolutionX)
        by = (bildausschnitt/resolutionY)
        for i in range(0,resolutionX+1):
            #print(i , "/" , resolutionX)
            for j in range(0,resolutionY+1):
                x = bi + bx * i
                y = bi + by * j
                vectorX = np.append(vectorX,x)
                vectorY = np.append(vectorY,y)
                e = self.gradient([x,y],Sn)
                vectorU = np.append(vectorU,-e[0])
                vectorV = np.append(vectorV,-e[1])
            
        plt.quiver(vectorX,vectorY,vectorU,vectorV, angles = 'xy')

#################################

##########################

if __name__ == '__main__':

    if len(sys.argv) > 1:
        fileName = sys.argv[1]
    else:
        fileName = "test"

    #set Seed
    np.random.seed(0)

    ### Define Mode of Operation ###
    gridSearch = False
    randomSearch = not gridSearch
    useDataSet =  True
    doRegression = False
    doClassificationWithInDistributionDetection = True
    doVolume = False
    getTime = False
    crossValidation = False

    showColorGradient = False
    showVectorField = False
    showPlots = False
    calculateUncertainyAtDataPoints = False

    ### Design Parameter ###
    featureDimentionInit = 100
    featureDimentionStep = 100
    sigmaInit = 1
    sigmaStep = 0
    alphaExponentInit = -1
    alphaExponentStep = 0
    betaInit = 0
    betaStep = 0
    threshholdfactorInit = 10
    threshholdfactorStep = 10
    slope = 1.0

    ### Define Variable Values ###
    VariableValue1 = 1 # one Graph per Value or Y-Axis: 0=featureDimention, 1=alpha, 2=sigma, 3=beta, 4=threshholdfactor1 5=threshholdfactor2 
    VariableValue2 = 0 # Value for X-Axis: 0=featureDimention, 1=alpha, 2=sigma, 3=beta, 4=threshholdfactor 5=threshholdfactor2 

    ### Sonstige Parameter ###
    bildausschnitt = 8

    ### Randomsearch Parameter ###
    folds = 1 # Cross validations
    i1 = 1 #Change Value1
    i2 = 1 #Change Value2
    i3 = 1 #Samples

    searchInterations = 400
    randomSearchOrders = 15
    samples = 1
    testSamples = 5

    ### Volume Parameters ###
    x1 = -1
    y1 = -1
    w = 2
    h = 2
    x2 = x1 + w
    y2 = y1 + h
    rect = x1,y1,w,h

    ### Import Dataset ###
    
    trainingPercentage = 0.7
    validationPercentage = 0.3
    testPercentage = 0.7

    from datasets import load_dataset
    if useDataSet:
        if doRegression:
            #dsName = "reg_num_house_16H" #22784 / 16
            #dsName = "reg_num_houses" #20640 / 8
            dsName = "reg_num_pol" #15000 / 26
            #dsName = "reg_num_wine_quality" #6497 / 11
        else:
            #dsName = "clf_num_bank-marketing" #10578 / 7
            dsName = "clf_num_california" #20634 / 8  
            #dsName = "clf_num_covertype" #566602 / 10 
            #dsName = "clf_num_credit" #16714 / 10
            #dsName = "clf_num_electricity" #38474 / 7
            #dsName = "clf_num_eye_movements" #7608 / 20
            #dsName = "clf_num_Higgs" #940160 / 24
            #dsName = "clf_num_house_16H" #13488 / 16
            #dsName = "clf_num_jannis" #57580 / 54
            #dsName = "clf_num_kdd_ipums_la_97-small" #
            #dsName = "clf_num_MagicTelescope" #13376 / 10
            #dsName = "clf_num_MiniBooNE" #72998 / 50
            #dsName = "clf_num_phoneme" #
            #dsName = "clf_num_pol" #10082 / 26
            #dsName = "clf_num_wine" #  

        ds = load_dataset("inria-soda/tabular-benchmark", dsName , split="train")
        #ds = ds.with_format("cupy")
        print(ds)

        labeledData = True

    #############################

    if VariableValue1 == VariableValue2:
        raise ValueError("VariableValue1 and VariableValue2 must be different")
    
    if gridSearch:
        if VariableValue1 == 0 or VariableValue2 == 0:
            featureDimentionInit = featureDimentionInit - featureDimentionStep
        if VariableValue1 == 1 or VariableValue2 == 1:
            alphaExponentInit = alphaExponentInit - alphaExponentStep
        if VariableValue1 == 2 or VariableValue2 == 2:
            sigmaInit = sigmaInit - sigmaStep
        if VariableValue1 == 3 or VariableValue2 == 3:
            betaInit = betaInit - betaStep
        if VariableValue1 == 4 or VariableValue2 == 4:
            threshholdfactorInit = threshholdfactorInit - threshholdfactorStep
    
    if useDataSet:
        inputDimention = ds.num_columns - 1
    else:
        inputDimention = 2
    figure = 0
    
    ### Generating Dataset ###

    dataset = np.array([])
    dataLabels = np.array([])

    trainingSet = np.array([])
    trainingLabels = np.array([])

    validationSet = np.array([])
    validationLabels = np.array([])

    testSet = np.array([])
    testSetScaled = np.array([])
    testLabels = np.array([])

    trainingClass1 = np.array([])
    trainingClass2 = np.array([])
    

    if useDataSet:
        dataset_size = ds.num_rows
        numberofFeatures = ds.num_columns - 1
    else:
        dataset_size = 1000
        numberofFeatures = 2

    trainingset_size = math.floor(dataset_size * trainingPercentage)
    restset_size = dataset_size - trainingset_size
    testset_size = math.floor(restset_size * testPercentage)
    validationset_size = restset_size - testset_size

    if not useDataSet:
        #this code generates different random datasets,
        # which one is geereated cen be selected by setting dataset_num:
        dataset_num = 0 #Data set number 0-8, e.g.  2:circle; 8:"three dots"

        # available dataets names:
        dsName =['generated','sigmoid', 'branch', 'circle', 'branch_varnoise', 'circle_varnoise', 'dualsine', 'square', 'eight', 'circledotted'][dataset_num]

        noise = 0.08
        
        labeledData = False 

        # Parameters for the 'generated' set
        numberOfClusters = 2
        varianceX = 0.2
        varianceY = 0.1
        offset = 0.5

        inputDimention = numberofFeatures

        if 'generated'==dsName:
            labeledData = True
            for i in range(0,dataset_size//numberOfClusters):
                #dataSet = np.append(dataSet,0,[np.random.normal(0,0.5) + np.abs(np.random.normal(0,0.5)), np.random.normal(0,1)])
                if doVolume:
                    dataLabels = np.append(dataLabels,1)
                    dataset = np.append(dataset,[np.random.uniform(x1,x2) , np.random.uniform(y1,y2)])
                else:
                    if numberofFeatures == 2:
                        dataset = np.append(dataset,[np.random.normal(0,varianceX)-offset , np.random.normal(0,varianceY)-offset])
                    elif numberofFeatures == 3:
                        dataset = np.append(dataset,[np.random.normal(0,varianceX)-offset , np.random.normal(0,varianceX), np.random.normal(0,varianceX)])
                    dataLabels = np.append(dataLabels,0)

            for i in range(0,dataset_size//numberOfClusters):
                if doVolume:
                    dataLabels = np.append(dataLabels,1)
                    dataset = np.append(dataset,[np.random.uniform(x1,x2) , np.random.uniform(y1,y2)])
                else:
                    if numberofFeatures == 2:
                        dataset = np.append(dataset,[np.random.normal(0,varianceX)+offset , np.random.normal(0,varianceY)+offset])
                    elif numberofFeatures == 3:
                        dataset = np.append(dataset,[np.random.normal(0,varianceX)+offset , np.random.normal(0,varianceX), np.random.normal(0,varianceX)])
                    dataLabels = np.append(dataLabels,1)

        if 'sigmoid'==dsName:
            X = np.linspace(-1,1,dataset_size)
            Y = 4 * X + 0.5
            Y = 1./(1+np.exp(-Y))
            Y = Y + noise * np.random.randn(dataset_size)
            X = np.concatenate([X[:,None],Y[:,None]], axis=1)
        elif 'branch'==dsName:
            #linear part
            x = np.linspace(-2,0,dataset_size-2*(dataset_size//3))
            y = x
            #first branch
            x1 = np.linspace(0,3,dataset_size//3)
            y1 = np.sqrt(x1)
            #second branch
            x2 = np.linspace(0,2,dataset_size//3)
            y2 = -np.sqrt(x2)
            #collect all parts
            X = np.concatenate([x, x1, x2])
            Y = np.concatenate([y, y1, y2])
            #add noise
            Y = Y + noise * np.random.randn(dataset_size)
            X = np.concatenate([X[:,None],Y[:,None]], axis=1)
        elif 'circle'==dsName:
            phi = np.linspace(0,np.pi*2,dataset_size)
            X = np.sin(phi)
            Y = np.cos(phi)
            Y = Y + noise * np.random.randn(dataset_size)
            X = X + noise * np.random.randn(dataset_size)
            X = np.concatenate([X[:,None],Y[:,None]], axis=1)
        elif 'branch_varnoise'==dsName:
            #linear part
            x = np.linspace(-2,0,dataset_size-2*(dataset_size//3))
            y = x  + noise * np.random.randn(x.shape[0])
            #first branch
            x1 = np.linspace(0,3,dataset_size//3)
            y1 = np.sqrt(x1) + noise* 4 * np.random.randn(x1.shape[0]);
            #second branch
            x2 = np.linspace(0,2,dataset_size//3)
            y2 = -np.sqrt(x2)  + noise * 0.5 * np.random.randn(x2.shape[0]);
            #collect all parts
            X = np.concatenate([x, x1, x2])
            Y = np.concatenate([y, y1, y2])
            #add noise
            Y = Y
            X = np.concatenate([X[:,None],Y[:,None]], axis=1)
        elif 'circle_varnoise'==dsName:
            phi = np.linspace(0,np.pi*2,dataset_size)
            X = np.sin(phi)
            Y = np.cos(phi)
            Y = Y + noise*(X+1) * np.random.randn(dataset_size)
            X = X + noise*(X+1) * np.random.randn(dataset_size)
            X = np.concatenate([X[:,None],Y[:,None]], axis=1)
        elif 'dualsine'==dsName:
            phi = np.linspace(0,np.pi*4,dataset_size)
            X = np.concatenate([phi,phi])
            Y = np.concatenate([np.sin(phi)+1.5,np.sin(phi)-1.5])
            Y = Y + noise * np.concatenate([np.random.randn(dataset_size), np.random.randn(dataset_size)])
            X = X + noise * np.random.randn(dataset_size*2)
            X = np.concatenate([X[:,None],Y[:,None]], axis=1)
        elif 'square'==dsName:
            phi = np.linspace(0,0.99,dataset_size)
            X = np.concatenate([phi])
            Y = (np.mod(np.floor(phi*4),2)-0.5)*1.0
            Y = Y + noise*2 * np.concatenate([np.random.randn(dataset_size)])
            X = X #+ noise * np.random.randn(dataset_size)
            X=X*4-2.0
            X = np.concatenate([X[:,None],Y[:,None]], axis=1)  
        elif 'eight'==dsName:
            phi = np.linspace(0,np.pi*2,dataset_size//2)
            X = np.sin(phi)+1
            Y = np.cos(phi)-1
            Y = Y + noise * np.random.randn(dataset_size//2)
            X = X + noise * np.random.randn(dataset_size//2)
            phi2 = np.linspace(0,np.pi*2,dataset_size-dataset_size//2)
            X2 = np.sin(phi)-1
            Y2 = np.cos(phi)+1
            Y2 = Y2 + noise*2 * np.random.randn(dataset_size-dataset_size//2)
            X2 = X2 + noise*2 * np.random.randn(dataset_size-dataset_size//2)
            X = np.concatenate([np.concatenate((X,X2))[:,None],np.concatenate((Y,Y2))[:,None]], axis=1)
        elif 'circledotted'==dsName:
            numdots=3
            phi = np.linspace(0,np.pi*2,dataset_size)
            phidist=2*np.pi/(numdots)
            phi = phi//phidist
            phi *= phidist
            X = np.sin(phi)
            Y = np.cos(phi)
            Y = Y + noise * np.random.randn(dataset_size)
            X = X + noise * np.random.randn(dataset_size)
            X = np.concatenate([X[:,None],Y[:,None]], axis=1)
        else:
            print("Error: no valid dataset defined!")

        if not dsName == 'generated':
            dataset = X
        
        ### Preprossesing Dataset ### 
        
        dataset = dataset.reshape(dataset_size,inputDimention)
        print("Dataset =", dataset)

        print("normalizeDataset")
        
        #scaler = preprocessing.QuantileTransformer().fit(dataSet)
        scaler = preprocessing.StandardScaler(with_mean=True,with_std=True).fit(dataset)
        #scaler = preprocessing.RobustScaler().fit(dataSet)
        print(scaler.mean_)
        print(scaler.scale_)
        x1 = x1 / scaler.scale_[0] - (scaler.mean_[0] / scaler.scale_[0])
        y1 = y1/ scaler.scale_[1] -(scaler.mean_[1] / scaler.scale_[1])
        w = w / scaler.scale_[0]
        h = h / scaler.scale_[1]
        rect = x1,y1,w,h
        dataset = scaler.transform(dataset)
        dataset = np.array(dataset)
        #print(dataSet)

        if labeledData:
            dataLabels = dataLabels.reshape(dataset_size,1)
        else:
            # if Data is unlabeled create placeholder labels
            dataLabels = np.zeros(dataset_size//2)
            dataLabels = np.append(dataLabels,np.ones(dataset_size//2))
            dataLabels = dataLabels.reshape(dataset_size,1)

        dataset = np.append(dataset,dataLabels,axis=1)
    
    if useDataSet:
        print("Constructing DataSet")
        for i in range(0,dataset_size):
            t = list(ds[i].values())
            if not doRegression:
                if i < dataset_size//2:
                    t[numberofFeatures] = 0
                else:
                    t[numberofFeatures] = 1
            dataset = np.append(dataset,np.array(t))

        dataset = dataset.reshape(dataset_size,ds.num_columns)

        class1Label	= 0
        class2Label = 1

    ################################

    # Global Functions
    
    def write_results_to_file(file_path, results):
        """Wirtes the results to File."""
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                results.tofile(file,sep=' ')
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")

    def do_train_validation_test_split(state):
        global dataset,trainingSet,restSet,validationSet,testSet, testSetScaled,trainingLabels,testLabels,validationLabels,trainingClass1,trainingClass2
        trainingSet, restSet =  model_selection.train_test_split(dataset,test_size=restset_size, random_state=state)
        print("Train test split")
        #print(trainingSet)

        validationSet, testSet =  model_selection.train_test_split(restSet,test_size=testset_size, random_state=state)

        #print("Validation split")
        #print(validationSet)
        #print("Test split")
        #print(testSet)

        trainingLabels = trainingSet[:,numberofFeatures]
        trainingSet = trainingSet[:,0:numberofFeatures]

        testLabels = testSet[:,numberofFeatures]
        testSetScaled = testSet[:,0:numberofFeatures]

        validationLabels = validationSet[:,numberofFeatures]
        validationSet = validationSet[:,0:numberofFeatures]

        counter1 = 0
        counter2 = 0
        trainingClass1 = np.array([])
        trainingClass2 = np.array([])
        for i in range(trainingset_size):
            if trainingLabels[i] == 0:
                trainingClass1 = np.append(trainingClass1,trainingSet[i])
                counter1 +=1
            else: 
                trainingClass2 = np.append(trainingClass2,trainingSet[i])
                counter2 +=1

        trainingClass1 = trainingClass1.reshape(counter1,numberofFeatures)
        trainingClass2 = trainingClass2.reshape(counter2,numberofFeatures)

    def do_test_split(state):
        global dataset,trainingSet,restSet,validationSet,testSet,trainingLabels,testLabels,validationLabels,trainingClass1,trainingClass2
        restSet,testSet  =  model_selection.train_test_split(dataset,test_size=testset_size, random_state=state)
        
        testLabels = testSet[:,numberofFeatures]
        testSet = testSet[:,0:numberofFeatures]
    
    def do_train_validation_split(state):
        global dataset,trainingSet,restSet,validationSet,testSet,testSetScaled,trainingLabels,testLabels,validationLabels,trainingClass1,trainingClass2
        validationSet,trainingSet =  model_selection.train_test_split(restSet,test_size=trainingset_size, random_state=state)

        trainingLabels = trainingSet[:,numberofFeatures]
        trainingSet = trainingSet[:,0:numberofFeatures]

        validationLabels = validationSet[:,numberofFeatures]
        validationSet = validationSet[:,0:numberofFeatures]

        if doRegression:
            l = trainingLabels.reshape(trainingset_size,1)
            t = np.append(trainingSet,l,axis=1)
            trainingSet,scaler = preprocess(t)

            l = validationLabels.reshape(validationset_size,1)
            v = np.append(validationSet,l,axis=1)
            validationSet = scale(v,scaler)

            l = testLabels.reshape(testset_size,1)
            t = np.append(testSet,l,axis=1)
            testSet = scale(t,scaler)

            validationLabels = validationSet[:,numberofFeatures]
            validationSet = validationSet[:,0:numberofFeatures]

            testLabels = testSet[:,numberofFeatures]
            testSet = testSet[:,0:numberofFeatures]

        else:
            trainingSet,scaler = preprocess(trainingSet)
            validationSet = scale(validationSet,scaler)
            testSetScaled = scale(testSet,scaler)

            counter1 = 0
            counter2 = 0
            trainingClass1 = np.array([])
            trainingClass2 = np.array([])
            for i in range(trainingset_size):
                if trainingLabels[i] == 0:
                    trainingClass1 = np.append(trainingClass1,trainingSet[i])
                    counter1 +=1
                else: 
                    trainingClass2 = np.append(trainingClass2,trainingSet[i])
                    counter2 +=1

            trainingClass1 = trainingClass1.reshape(counter1,numberofFeatures)
            trainingClass2 = trainingClass2.reshape(counter2,numberofFeatures)

    def preprocess(set):
        ### Preprossesing Dataset ### 
        print("preprocess")
        #scaler = preprocessing.QuantileTransformer().fit(dataset)
        scaler = preprocessing.StandardScaler(with_mean=True,with_std=True).fit(set)
        #scaler = preprocessing.RobustScaler().fit(dataSet)
        set = scaler.transform(set)
        set = np.array(set)
        return set,scaler
    
    def scale(set,scaler):
        set = scaler.transform(set)
        set = np.array(set)
        return set

    #################################

    # Split into Train Validation and Testsets #
    if useDataSet:
        if not crossValidation:
            do_train_validation_test_split(0)
        else:
            do_test_split(0)
    else:
        trainingSet = dataset[:,0:numberofFeatures]

    ### Calculation ###
    counter = 0

    results = np.array([])

    featureDimentionTemp = featureDimentionInit
    alphaExponentTemp = alphaExponentInit
    sigmaTemp = sigmaInit
    betaTemp = betaInit
    threshholdfactorTemp = threshholdfactorInit
    threshholdfactor1Temp = threshholdfactorInit
    threshholdfactor2Temp = threshholdfactorInit

    if randomSearch:
        print("RandomSearch")
        searchDimention = True
        searchSigma = False
        searchAlpha = True
        searchBeta = False
        searchTreshhold = False

        scoresAtIterations = np.array([])
        dimentionsAtIterations = np.array([])
        sigmasAtIterations = np.array([])
        alphaExponentsAtIterations = np.array([])
        betasAtIterations = np.array([])
        threshholdfactors1AtIterations = np.array([])
        threshholdfactors2AtIterations = np.array([])

        for h in range(randomSearchOrders):

            if crossValidation:
                do_train_validation_split(h)

            bestScoreAvg = 0
            bestScoresAvg = np.array([])
            bestScoreMax = 0
            bestScoresMax = np.array([])
            bestScoreMin = 0
            bestScoresMin = np.array([])
            bestParams = np.array([])
            
            bestDimention = 0
            bestSigma = 0
            bestAlphaexponent = 0
            bestBeta = 0
            bestThreshholdfactor1 = 0
            bestThreshholdfactor2 = 0
               
            for i in range(searchInterations):
                if searchDimention:
                    featureDimentionTemp = np.random.randint(10,1000)
                    if i == 0:
                        featureDimentionTemp = featureDimentionInit
                if searchSigma:
                    sigmaTemp = np.random.choice(np.array([0.5, 1, 2, 5, 10, 20]),1)[0]
                    if sigmaTemp <= 0:
                        sigmaTemp = 0.01
                    if i == 0:
                        sigmaTemp = sigmaInit
                if searchAlpha:
                    alphaExponentTemp = np.random.uniform(10,-20)
                    if i == 0:
                        alphaExponentTemp = alphaExponentInit
                if searchBeta:
                    betaTemp = np.random.uniform(0,-10)
                    if i == 0:
                       betaTemp = 0
                if searchTreshhold:
                    threshholdfactor1Temp = np.random.randint(1,100)
                    threshholdfactor2Temp = np.random.randint(1,100)
                    if i == 0:
                        threshholdfactor1Temp = threshholdfactorInit
                        threshholdfactor2Temp = threshholdfactorInit

                counter = counter + 1
                print(counter,"/",searchInterations*randomSearchOrders)
                print("--- %s seconds ---" % (time.time() - start_time))

                m = EpistemicUncertaintyModel(inputDimention,featureDimentionTemp,alphaExponentTemp,sigmaTemp,betaTemp,threshholdfactor1Temp)
                accTemp = np.average(m.evaluateModel(False))
                print(accTemp)

                scoreAvg = np.average(accTemp)
                if scoreAvg > bestScoreAvg:
                    bestScoreAvg = scoreAvg
                    bestDimention = featureDimentionTemp
                    bestSigma = sigmaTemp
                    bestAlphaexponent = alphaExponentTemp
                    bestBeta = betaTemp
                    bestThreshholdfactor1 = threshholdfactor1Temp
                    bestThreshholdfactor2 = threshholdfactor2Temp
                bestScoresAvg = np.append(bestScoresAvg,bestScoreAvg)

                m = EpistemicUncertaintyModel(inputDimention,bestDimention,bestAlphaexponent,bestSigma,bestBeta,bestThreshholdfactor1)
                accTest = np.average(m.evaluateModel(True))
                print(accTest)
                results = np.append(results,np.array([accTest,featureDimentionTemp,sigmaTemp, alphaExponentTemp, betaTemp, threshholdfactorTemp, threshholdfactorTemp, trainingset_size, i]),axis=0)

                scoreMax = np.amax(accTemp)
                if scoreMax > bestScoreMax:
                    bestScoreMax = scoreMax
                bestScoresMax = np.append(bestScoresMax,bestScoreMax)

                scoreMin = np.amin(accTemp)
                if scoreMin > bestScoreMin:
                    bestScoreMin = scoreMin
                bestScoresMin = np.append(bestScoresMin,bestScoreMin)

                bestParams = np.append(bestParams,np.array([bestDimention,bestSigma,bestAlphaexponent,bestBeta]))

                scoresAtIterations = np.append(scoresAtIterations,bestScoreAvg)
                dimentionsAtIterations = np.append(dimentionsAtIterations,bestDimention)
                sigmasAtIterations = np.append(sigmasAtIterations,bestSigma)
                alphaExponentsAtIterations = np.append(alphaExponentsAtIterations,bestAlphaexponent)
                betasAtIterations = np.append(betasAtIterations,bestBeta)
                threshholdfactors1AtIterations = np.append(threshholdfactors1AtIterations,bestThreshholdfactor1)
                threshholdfactors2AtIterations = np.append(threshholdfactors2AtIterations,bestThreshholdfactor2)

        bestParams = bestParams.reshape(searchInterations,4)
        print(bestParams)

        scoresAtIterations = scoresAtIterations.reshape(randomSearchOrders,searchInterations)
        scoresAtIterations = np.transpose(scoresAtIterations)
        dimentionsAtIterations = dimentionsAtIterations.reshape(randomSearchOrders,searchInterations)
        dimentionsAtIterations = np.transpose(dimentionsAtIterations)
        sigmasAtIterations = sigmasAtIterations.reshape(randomSearchOrders,searchInterations)
        sigmasAtIterations = np.transpose(sigmasAtIterations)
        alphaExponentsAtIterations = alphaExponentsAtIterations.reshape(randomSearchOrders,searchInterations)
        alphaExponentsAtIterations = np.transpose(alphaExponentsAtIterations)
        betasAtIterations = betasAtIterations.reshape(randomSearchOrders,searchInterations)
        betasAtIterations = np.transpose(betasAtIterations)
        threshholdfactors1AtIterations = threshholdfactors1AtIterations.reshape(randomSearchOrders,searchInterations)
        threshholdfactors1AtIterations = np.transpose(threshholdfactors1AtIterations)
        threshholdfactors2AtIterations = threshholdfactors2AtIterations.reshape(randomSearchOrders,searchInterations)
        threshholdfactors2AtIterations = np.transpose(threshholdfactors2AtIterations)

        bestDimention = int(np.average(dimentionsAtIterations[-1]))
        bestSigma = np.average(sigmasAtIterations[-1])
        bestAlphaexponent = np.average(alphaExponentsAtIterations[-1])
        bestBeta = np.average(betasAtIterations[-1])
        bestThreshholdfactor1 = np.average(threshholdfactors1AtIterations[-1])
        bestThreshholdfactor2 = np.average(threshholdfactors2AtIterations[-1])

        featureDimentionTemp = bestDimention
        sigmaTemp = bestSigma
        alphaExponentTemp = bestAlphaexponent
        betaTemp = bestBeta
        threshholdfactor1Temp = bestThreshholdfactor1
        threshholdfactor2Temp = bestThreshholdfactor2

        print(scoresAtIterations)
        print(dimentionsAtIterations)
        print(alphaExponentsAtIterations)
        print(sigmasAtIterations)
        print(betasAtIterations)

        print(bestDimention)
        print(bestSigma)
        print(bestAlphaexponent)
        print(bestBeta)
        print(bestThreshholdfactor1)
        print(bestThreshholdfactor1)

    if gridSearch:

        xAxis3D = np.array([])
        yAxis3D = np.array([])
        zAxis3D = np.array([])

        for h in range(folds):

            if useDataSet:
                do_train_validation_split(h)

                featureDimentionTemp = featureDimentionInit
                alphaExponentTemp = alphaExponentInit
                sigmaTemp = sigmaInit
                betaTemp = betaInit
                threshholdfactorTemp = threshholdfactorInit
                threshholdfactor1Temp = threshholdfactorInit
                threshholdfactor2Temp = threshholdfactorInit

            for i in range(i1):

                xAxis = np.array([])
                yAxis = np.array([])
                
                AADP1= np.array([])
                AVDP1 = np.array([])
                AADP2 = np.array([])
                AVDP2 = np.array([])
                AAGD = np.array([])
                AVGD = np.array([])
                
                AverageVolume = np.array([])
                AverageClassificationError = np.array([])

                if VariableValue1 == 0:
                    featureDimentionTemp = featureDimentionTemp + featureDimentionStep
                elif VariableValue1 == 1:
                    alphaExponentTemp = alphaExponentTemp + alphaExponentStep
                elif VariableValue1 == 2:
                    sigmaTemp = sigmaTemp + sigmaStep
                elif VariableValue1 == 3:
                    betaTemp = betaTemp + betaStep
                elif VariableValue1 == 4:
                    threshholdfactorTemp = threshholdfactorTemp + threshholdfactorStep
                elif VariableValue1 == 5:
                    threshholdfactor2Temp = threshholdfactor2Temp + threshholdfactorStep

                if VariableValue2 == 0:
                    featureDimentionTemp = featureDimentionInit
                elif VariableValue2 == 1:
                    alphaExponentTemp = alphaExponentInit
                elif VariableValue2 == 2:
                    sigmaTemp = sigmaInit
                elif VariableValue2 == 3:
                    betaTemp = betaInit
                elif VariableValue2 == 4:
                    threshholdfactorTemp = threshholdfactorInit
                elif VariableValue2 == 5:
                    threshholdfactor2Temp = threshholdfactorInit

                for j in range(0,i2):

                    #xAxis
                    if VariableValue2 == 0:
                        featureDimentionTemp = featureDimentionTemp + featureDimentionStep
                        xAxis = np.append(xAxis,featureDimentionTemp)
                        xAxis3D = np.append(xAxis3D,featureDimentionTemp)
                    elif VariableValue2 == 1:
                        alphaExponentTemp = alphaExponentTemp + alphaExponentStep
                        xAxis = np.append(xAxis,alphaExponentTemp)
                        xAxis3D = np.append(xAxis3D,alphaExponentTemp)
                    elif VariableValue2 == 2:
                        sigmaTemp = sigmaTemp + sigmaStep
                        xAxis = np.append(xAxis,sigmaTemp)
                        xAxis3D = np.append(xAxis3D,sigmaTemp)
                    elif VariableValue2 == 3:
                        betaTemp = betaTemp + betaStep
                        xAxis = np.append(xAxis,betaTemp)
                        xAxis3D = np.append(xAxis3D,betaTemp)
                    elif VariableValue2 == 4:
                        threshholdfactorTemp = threshholdfactorTemp + threshholdfactorStep
                        xAxis = np.append(xAxis,threshholdfactorTemp)
                        xAxis3D = np.append(xAxis3D,threshholdfactorTemp)
                    elif VariableValue2 == 5:
                        threshholdfactor2Temp = threshholdfactor2Temp + threshholdfactorStep
                        xAxis = np.append(xAxis,threshholdfactor2Temp)
                        xAxis3D = np.append(xAxis3D,threshholdfactor2Temp)
                    #yAxis
                    if VariableValue1 == 0:
                        yAxis = np.append(yAxis,featureDimentionTemp)
                        yAxis3D = np.append(yAxis3D,featureDimentionTemp)
                    elif VariableValue1 == 1:
                        yAxis = np.append(yAxis,alphaExponentTemp)
                        yAxis3D = np.append(yAxis3D,alphaExponentTemp)
                    elif VariableValue1 == 2:
                        yAxis = np.append(yAxis,sigmaTemp)
                        yAxis3D = np.append(yAxis3D,sigmaTemp)
                    elif VariableValue1 == 3:
                        yAxis = np.append(yAxis,betaTemp)
                        yAxis3D = np.append(yAxis3D,betaTemp)
                    elif VariableValue1 == 4:
                        yAxis = np.append(yAxis,threshholdfactorTemp)
                        yAxis3D = np.append(yAxis3D,threshholdfactorTemp)
                    elif VariableValue1 == 5:
                        yAxis = np.append(yAxis,threshholdfactor2Temp)
                        yAxis3D = np.append(yAxis3D,threshholdfactor2Temp)

                    AveragesOfEpistemicUncertaintyAtDataPoints1 = np.array([])
                    VariancesOfEpistemicUncertaintyAtDataPoints1 = np.array([])
                    AveragesOfEpistemicUncertaintyAtDataPoints2 = np.array([])
                    VariancesOfEpistemicUncertaintyAtDataPoints2 = np.array([])
                    accVolume = np.array([])

                    classificationAccuracy = np.array([])
                    
                    for k in range(0,i3):
                        counter = counter + 1
                        print(counter,"/",i1*i2*i3*folds)

                        #Create Model
                        time1 = (time.time() - start_time)
                        m = EpistemicUncertaintyModel(inputDimention,featureDimentionTemp,alphaExponentTemp,sigmaTemp,betaTemp,threshholdfactorTemp,threshholdfactor2Temp)
                        time2 = (time.time() - start_time)
                        deltaTime = time2 - time1
                        deltaTime = m.getDeltaTime()

                        ### Compute epistemic uncertainty at data points ###
                        epistemicUncertaintyAtDataPoints1 = np.array([])
                        epistemicUncertaintyAtDataPoints2 = np.array([])
                        if calculateUncertainyAtDataPoints:
                            pool = mp.Pool(mp.cpu_count())
                            if not doClassificationWithInDistributionDetection:
                                #Berechene Epistemische Unsicherheit an datenpunkten
                                print("Calculating epistemic Uncertainty at Data Points")
                                for i in range(0,trainingset_size):
                                    e = m.epistemic(m.features[i],m.Sn)
                                    epistemicUncertaintyAtDataPoints1 = np.append(epistemicUncertaintyAtDataPoints1,0,e)
                                    #print("e =",e)
                                    #print(i)
                                print("--- %s seconds ---" % (time.time() - start_time))
                                epistemicUncertaintyAfterGradientDecent = np.array([])

                            if doClassificationWithInDistributionDetection:
                                #Berechene Epistemische Unsicherheit an datenpunkten
                                print("Calculating epistemic Uncertainty at Data Points")
                                TASKS1 = [(m.features1[i],m.Sn1)for i in range(trainingset_size//2)]
                                TASKS1 = TASKS1 + [(m.features2[i],m.Sn1)for i in range(trainingset_size//2)]
                                TASKS2 = [(m.features1[i],m.Sn2)for i in range(trainingset_size//2)]
                                TASKS2 = TASKS2 + [(m.features2[i],m.Sn2)for i in range(trainingset_size//2)]
                                print("--- %s seconds ---" % (time.time() - start_time))
                                epis1 = pool.starmap(m.epistemic,TASKS1, chunksize=100)
                                epistemicUncertaintyAtDataPoints1 = np.array(epis1)
                                print("--- %s seconds ---" % (time.time() - start_time))
                                epis2 = pool.starmap(m.epistemic,TASKS2, chunksize=100)
                                epistemicUncertaintyAtDataPoints2 = np.array(epis2)
                                print("--- %s seconds ---" % (time.time() - start_time))
                            
                            AveragesOfEpistemicUncertaintyAtDataPoints1 = np.append(AveragesOfEpistemicUncertaintyAtDataPoints1,np.average(epistemicUncertaintyAtDataPoints1))
                            VariancesOfEpistemicUncertaintyAtDataPoints1 = np.append(VariancesOfEpistemicUncertaintyAtDataPoints1,np.var(epistemicUncertaintyAtDataPoints1))

                            if doClassificationWithInDistributionDetection:
                                AveragesOfEpistemicUncertaintyAtDataPoints2 = np.append(AveragesOfEpistemicUncertaintyAtDataPoints2,np.average(epistemicUncertaintyAtDataPoints2))
                                VariancesOfEpistemicUncertaintyAtDataPoints2 = np.append(VariancesOfEpistemicUncertaintyAtDataPoints2,np.var(epistemicUncertaintyAtDataPoints2))

                            AverageUncertainty1 = np.average(epistemicUncertaintyAtDataPoints1)
                            AverageUncertainty2 = np.average(epistemicUncertaintyAtDataPoints2)

                        if getTime:
                            acc = deltaTime
                        else:
                            #evaluate Model
                            acc = m.evaluateModel(False,showColorGradient)
                        classificationAccuracy = np.append(classificationAccuracy,acc)

                    results = np.append(results,np.array([np.average(classificationAccuracy),featureDimentionTemp,sigmaTemp, alphaExponentTemp, betaTemp, threshholdfactorTemp, threshholdfactor2Temp, trainingset_size,h]),axis=0)
                    
                    if calculateUncertainyAtDataPoints:
                        AADP1 = np.append(AADP1,np.average(AveragesOfEpistemicUncertaintyAtDataPoints1))
                        AVDP1 = np.append(AVDP1,np.average(VariancesOfEpistemicUncertaintyAtDataPoints1))
                        if doClassificationWithInDistributionDetection:
                            AADP2 = np.append(AADP2,np.average(AveragesOfEpistemicUncertaintyAtDataPoints2))
                            AVDP2 = np.append(AVDP2,np.average(VariancesOfEpistemicUncertaintyAtDataPoints2))
                    
                    AverageClassificationError = np.append(AverageClassificationError,np.average(classificationAccuracy))

    print("--- %s seconds ---" % (time.time() - start_time))

    if gridSearch:
        results = results.reshape(i1*i2*folds,9)

    if randomSearch:
        results = results.reshape(randomSearchOrders,searchInterations,9)

    ("Results: ")
    print(results)
    write_results_to_file(fileName + '.txt', results)

    if showPlots:
        plt.show()