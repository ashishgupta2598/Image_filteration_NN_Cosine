
import torchvision.models as models
import torch
from torch import nn
import PIL
import os
from torchvision import transforms
import csv
import pickle

trans=transforms.Compose([
    transforms.Resize((250)),
    transforms.CenterCrop((250,150)),
    transforms.ToTensor()])
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier=vgg16.classifier[:1]

dire=os.listdir('./out1')
########TAKING 34770----------len(dire)=34769
#vectors=torch.empty((len(dire),4096))
#dict1={}
#print(len(dire))
#exit()
print("I am Starting")
for j in range(114):
    vectors=torch.empty((305,4096))
    for i in range(0,305):
        
        try:
            img=PIL.Image.open('./out1/'+dire[i])
            inp=trans(img) 
            sh=inp.shape
            inp=inp.view(1,sh[0],sh[1],sh[2])
            out=vgg16(inp)
            vectors[i]=out[0]
            #l1={dire[i]:out}    
            #dict1.update(l1)
        
        except Exception as e :
            print(e)
            pass
    path='tensors_cosine/tensor_'+str(j)+'.pt'        
    torch.save(vectors,path)        
    #ict1.update(l1)
               
    print(j)
#torch.save(vectors,'tensor.pt')    
# f = open("latent_vectors.pkl","wb")
# pickle.dump(dict1,f)
# f.close()    
#torch.save('latent_vector.pt',vectors)    
    