import numpy as np
from os import listdir, getcwd
from os.path import isfile, join


def speed(delta_t, delta_x, delta_y):
    return np.sqrt((delta_x / delta_t) ** 2 + (delta_y / delta_t) ** 2)


def readData(filename):
    data = np.loadtxt(filename)
    if "forsok02" in filename or "forsok08" in filename:
        delta_t = data[1][0] - data[0][0]
    else:
        delta_t = (data[1][0] - data[0][0]) / 4
    delta_x = data[1][1] - data[0][1]
    delta_y = data[1][2] - data[0][2]
    return speed(delta_t, delta_x, delta_y)

def l_hopp(v0, h_hopp, g):
    return v0*np.sqrt(2*h_hopp/g)


def main():
    mypath = getcwd() + "/resources/"
    runs = 0
    
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)
    speed_on_edge = np.array([])
    for txtFile in onlyfiles:
        runs += 1
        value = readData(mypath + txtFile)
        speed_on_edge = np.append(speed_on_edge, value)
        print(txtFile + ": " + str(value))

    v0 = np.average(speed_on_edge)
    v0_standard_derivation = np.std(speed_on_edge)
    g = 9.8214676
    h_hopp = 0.093

    print("")
    print("v0: " + str(v0))
    print("standard derivation v0: " + str(v0_standard_derivation))
    print("standard error v0: " + str(v0_standard_derivation/np.sqrt(runs)))

    print("l_hopp: " + str(l_hopp(v0, h_hopp, g)))

if __name__ == "__main__":
    main()
