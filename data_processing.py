import os
import numpy as np
import pandas as pd
from pathlib import Path

def data_convert(csv_root, npy_root):
    csv_file_list = os.listdir(csv_root)
    if (Path(npy_root).is_dir()==False):
        os.mkdir(npy_root)

    # make dict.txt
    if(os.path.isfile("./dict.txt")):
        os.remove("./dict.txt")

    # make npy dir
    for i in range(len(csv_file_list)):
        temp = (npy_root+csv_file_list[i][0:-4]).replace(" ", "")
        temp = temp.replace("'","")
        temp = temp.replace("[","")
        temp = temp.replace("]","")
        if (Path(temp).is_dir()==False):
            os.mkdir(temp)
        else:
            pass
        temp = temp.split("/")[-1]

        # save as .npy
        data = np.array(pd.read_csv(csv_root + csv_file_list[i]))[:, 1:].T
        for j in range(len(data)):
            signal = data[j]
            np.save(npy_root+temp+'/'+str(i)+"_"+temp+"_"+str(j)+".npy", signal)
        with open("./dict.txt", "a") as f:
            f.write(temp+","+str(i)+"\n")
        print("current : ", i, "/", len(csv_file_list), "  ", temp)

if __name__ == "__main__":
    data_convert("./dataset8/csv/train/", "./dataset8/npy/train/")
    data_convert("./dataset8/csv/test/", "./dataset8/npy/test/")