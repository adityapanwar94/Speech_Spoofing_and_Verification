import numpy as np
from scipy.io import wavfile as sp
import re
from python_speech_features import mfcc
import soundfile as sf

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

#initializing lists to hold mfcc features and transcipt encoded form
t = []
w = []
for j in range(225, 376):
    mdir = 's2tdata/wav48/p{0:0=3d}/p{0:0=3d}'.format(j, j)
    mdirt = 's2tdata/txt/p{0:0=3d}/p{0:0=3d}'.format(j, j)
    print(j)
    for i in range(1, 501):
        dir = mdir + '_{0:0=3d}.wav'.format(i)
        dirt = mdirt + '_{0:0=3d}.txt'.format(i)
        #print(dir)
        try:
            x, rate = sf.read(dir)
            features = mfcc(x, rate)
            w.append(features)

            with open(dirt, 'r') as f:
                # Only the last line is necessary
                line = f.readlines()[-1]
                # removing everything other than alphabets and spaces
                regex = re.compile('[^a-zA-Z ]')
                line = regex.sub('', line)

                # Get only the words between [a-z] and replace period for none
                original = ' '.join(line.strip().lower().split(' ')[:]).replace('.', '')
                targets = original.replace(' ', '  ')
                targets = targets.split(' ')

            # Adding blank label
            targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

            # Transform char into index
            targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])
            t.append(targets)
        except:
            continue


w = np.asarray(w)
t = np.asarray(t)
#printing the length to verify both the matrix are the same length
print(len(w))
print(len(t))
#saving the whole matrix of data
np.save('convmodel/train_input', w)
np.save('convmodel/train_label', t)
print('saved!')
