import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def read_and_print_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            contents = np.fromfile(file,sep=' ')
            print("File Contents:\n")
            print(contents)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return contents

def plot(array,label_in=''):

    if gridsearch:
        #entries = len(array) // 8
        #array = array.reshape(entries,8)
        entries = len(array) // 9
        array = np.split(array,entries)
        array = np.array(array) #7500 / 9

        index = np.argsort(array[:,-1], axis=0)
        array = array[index]
        array = np.split(array,crossFolds)
        array = np.array(array) # 5 / 1500 / 9
        
        if multiDim:
            #for j in range(5):
            index = np.argsort(array[:,:,3], axis=1)
            for i in range(crossFolds):
                array[i] = array[i,index[i]]
            
            array = np.array(np.split(array,50,axis=1))

            for i in range(50):
                for j in range(crossFolds):
                    index = np.argsort(array[i,j,:,4], axis=0)
                    array[i,j] = array[i,j,index]

            xAxis = np.linspace(-20,10,30) 
            xAxis = np.linspace(1,50,50)
        else:
            for j in range(crossFolds):
                    index = np.argsort(array[j,:,2], axis=0)
                    array[j] = array[j,index]
    else:
        entries = len(array) // 9
        array = np.split(array,entries)
        array = np.array(array)
        index = np.argsort(array[:,-1], axis=0)
        array = array[index]
        array = np.split(array,searchIterations)
        array = np.array(array)
        xAxis = np.linspace(1,searchIterations,searchIterations)
    
    if gridsearch:
        plt.figure(800)
        title = "Berechnungszeit"
        #plt.title(title)
        #plt.yscale("log")
        plt.xlabel("Sigma")
        plt.ylabel("Genauigkeit")
        #plt.xlim(1,searchIterations)
        #df = pd.DataFrame(array[:,0])
        #y = np.array(df[0].rolling(100).mean())
        #plt.plot(array[:,1],array[:,0], label=label_in)#,color='lightblue'
        if multiDim:
            #for i in range(24):
            #    #j = 2*i+4
            #    j=i
            #   plt.plot(xAxis,np.average(array[:,:,j,0],axis=1), label=array[0,0,j,3]) 
                
            fig = plt.figure(1000)
            ax = fig.add_subplot(projection='3d')
            c = 0
            ax.scatter(array[:,c,:,3],array[:,c,:,4],np.amax(array[:,:,:,0],axis=1), 'o')
            ax.set_xlabel('Alpha Exponent')
            ax.set_ylabel('Beta Exponent')
            ax.set_zlabel('Genauigkeit')
            print(np.amax(array[:,:,:,0],axis=1))
            plt.legend()
        else:
            plt.plot(array[0,:,2],np.average(array[:,:,0],axis=0), label=label_in)#,color='lightblue'
            #plt.ylim(0.72,0.87)
            #plt.xlim(-15,20)
            #plt.legend()
    else:
        fig = plt.figure(1150)
        ax = fig.add_subplot()
        #ax.plot(np.amax(scoresAtIterations,axis=1), label='Max')
        #ax.plot(np.amin(scoresAtIterations,axis=1), label='Min')
        ax.fill_between(xAxis,np.amax(array[:,:,0],axis=1),np.average(array[:,:,0],axis=1),color=(0.1, 0.2, 0.5, 0.3))
        ax.fill_between(xAxis,np.amin(array[:,:,0],axis=1),np.average(array[:,:,0],axis=1),color=(0.1, 0.2, 0.5, 0.3))
        ax.plot(xAxis,np.average(array[:,:,0],axis=1), label=label_in) 
        plt.xlabel('Iteration')
        plt.ylabel("Genauigkeit")
        plt.xscale("log")
        plt.xlim(1,500)
        #plt.ylim(0.55,1.0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        #plt.legend()

if __name__ == "__main__": 
    plot0 = False
    plot1 = False
    plot2 = False
    plot3 = False
    referenz = True
    gridsearch = False
    multiDim = False
    searchIterations = 400
    crossFolds = 10
    if plot0:
        file_path = "test.txt"  # Replace with your file path
        array = read_and_print_file(file_path)
        plot(array,'normal')
    if plot1:
        file_path = "test1.txt"  # Replace with your file path
        array = read_and_print_file(file_path)
        plot(array,'uniform')
    if plot2:
        file_path = "test2.txt"  # Replace with your file path
        array = read_and_print_file(file_path)
        plot(array,'diskret')
    if plot3:
        file_path = "test3.txt"  # Replace with your file path
        array = read_and_print_file(file_path)
        plot(array,'kein Bias')
    if referenz:
        file_path = "referenz.txt"  # Replace with your file path
        array = read_and_print_file(file_path)
        plot(array,'threshhold')

    plt.show()