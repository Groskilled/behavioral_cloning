import pandas as pd
import numpy as np
from scipy.ndimage import imread
from model import nvidia

lines = pd.read_csv('../linux_sim/saved_videos/driving_log.csv')
images = []
angles = []
for i in range(len(lines)):
    im = imread('../linux_sim/saved_videos/IMG/' + lines.iloc[i]['center_image'].split('/')[-1])
    images.append(im)
    images.append(np.flip(im,1))
    angles.append(lines.iloc[i]['steering_angle'])
    angles.append(-lines.iloc[i]['steering_angle'])

    im = imread('../linux_sim/saved_videos/IMG/' + lines.iloc[i]['left_image'].split('/')[-1])
    images.append(im)
    angles.append(lines.iloc[i]['steering_angle'] + .25)

    im = imread('../linux_sim/saved_videos/IMG/' + lines.iloc[i]['right_image'].split('/')[-1])
    images.append(im)
    angles.append(lines.iloc[i]['steering_angle'] - .25)

images=np.array(images)
angles=np.array(angles)
model = nvidia(images[0].shape)
model.compile(loss='mse', optimizer='adam')
model.fit(images,angles,validation_split=0.2,shuffle=True, batch_size=4, epochs=3)
model.save('weights.h5')
