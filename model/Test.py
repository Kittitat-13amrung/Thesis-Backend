from DataGenerator import DataGenerator
import os
import pandas as pd
import numpy as np

batch_size=128
epochs=10
con_win_size = 9
spec_repr="c"
data_path= os.getcwd() + "/data/spec_repr/"
id_file="id.csv"
save_path="saved/"
halfwin = con_win_size // 2

label_dim = (6,21)
csv_file = data_path + id_file
list_IDs = list(pd.read_csv(os.getcwd()+'/data/spec_repr/id.csv', header=None)[0])
X_dim = (batch_size, 192, con_win_size, 1)
y_dim = (batch_size, label_dim[0], label_dim[1])
X = np.empty(X_dim)
y = np.empty(y_dim)
    
# determine filename
data_dir = data_path + spec_repr + "/"
frame_idx = int('05_SS3-98-C_solo_1'.split("_")[-1])

# load a context window centered around the frame index
loaded = np.load(data_dir + '05_SS3-98-C_solo.npz')
full_x = np.pad(loaded["repr"], [(halfwin,halfwin), (0,0)], mode='constant')
sample_x = full_x[frame_idx : frame_idx + con_win_size]
X[0,] = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)

# Store label
y[0,] = loaded["labels"][frame_idx]

print(loaded['labels'][2])

# DataGenerator(train, 
#             data_path=data_path, 
#             batch_size=batch_size, 
#             shuffle=True,
#             spec_repr=spec_repr, 
#             con_win_size=con_win_size)

# train = []
# for ID in list_IDs:
#     guitarist = int(ID.split("_")[0])
#     if guitarist == 2:
#         continue
#     else:
#         train.append(ID)

# training_generator = DataGenerator(train, 
#                                     data_path=data_path, 
#                                     batch_size=batch_size, 
#                                     shuffle=True,
#                                     spec_repr=spec_repr, 
#                                     con_win_size=con_win_size)