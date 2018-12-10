import sys
sys.path.append('/home/se26956/projects/')
import torch
from ontological_audio_embeddings.src.models.PatchGAN import Generator, Discriminator
import numpy as np
#from torchsummary import summary

#DATA_DIR = '/home/akasha/projects/ontological_audio_embeddings/data/preprocessed/rawAudioSetv2/'
#sample_file1 = "audio_ZZuL34BfdYY.npy"
#sample_file2 = "audio_oyuoIs62IPc.npy"


#x1 = np.expand_dims(np.expand_dims(np.load(DATA_DIR+sample_file1), axis=0), axis=0)
#max_x = np.max(x1)
#min_x = np.min(x1)
#x1 = x1/(max_x - min_x)
x1 = torch.rand((1, 2, 160083))

x1 = torch.tensor(x1)
generator = Generator(input_dim=1)
discriminator = Discriminator(input_dim=2)

#summary(generator, (1, x1.size(1)))

o = discriminator(x1)
print(o.size())

x = torch.rand((1, 1, 160083))
o = generator.getEmbedding(x)
print(o.size())

