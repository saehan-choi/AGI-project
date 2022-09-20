import torch
import torch.nn as nn
import torch.optim as optim

from model import *

class CFG:
    lr = 3e-4
    loss_threshold = 0.001
    
# loss값 이상하게 나올수도있으니 마지막으로 체크할것.

if __name__ == "__main__":
    while True:
        sentance = torch.randn(100)

        understandNet = NeuralNet()
        talkNet = NeuralNet()

        criterion = nn.MSELoss()
        optimizer_under = optim.Adam(understandNet.parameters(), lr=CFG.lr)
        optimizer_talk = optim.Adam(talkNet.parameters(), lr=CFG.lr)


        with torch.no_grad():
            talkNet.eval()
            talk = talkNet(sentance)

        understandNet.eval()
        understand = understandNet(sentance)
        
        understand_loss = criterion(understand, talk)
        
        print(f'understand_loss : {understand_loss}')
        
        if understand_loss < CFG.loss_threshold:
            pass
        
        else:
            understandNet.train()
            while True:
                understand = understandNet(sentance)
                
                understand_loss = criterion(understand, talk)
                
                optimizer_under.zero_grad()  
                understand_loss.backward()
                
                optimizer_under.step()
                with torch.no_grad():
                    understandNet.eval()
                    understand_loss = criterion(understand, talk)
                    print(understand_loss)
                    if understand_loss < CFG.loss_threshold:
                        break
                
