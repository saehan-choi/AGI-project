import torch
import torch.nn as nn
import torch.optim as optim


from model import *

import pandas as pd

class CFG:
    lr = 3e-4
    # loss_threshold = 0.001
    # loss_threshold = 0.0000001
    loss_threshold = 0.001
    # loss_threshold = 0.0000000001    -> 너무느림
    weights_save_path = './weights/'
    data_path = './data/df.csv'

def random_fill_zero(tensor):
    # tensor = torch.masked_fill(tensor, torch.tensor([False,  True, False, False, False, False,  True,  True,  True, False]),0)
    tensor = torch.masked_fill(tensor, torch.rand(100)>0.99, 0)
    # masking하면 확실히 이해하는데에 시간이 더 걸리네

    return tensor

def write_report(result):
    f = open('report.txt', 'a')
    f.write(str(result)+'\n')


def make_weights(model, weights_name):
    torch.save(model.state_dict(), CFG.weights_save_path+weights_name)

def make_dataframe(input_tensor, output_tensor):

    try:
        df = pd.read_csv(CFG.data_path)
        df.to_csv(CFG.data_path, index=False)
        
    except:
        df = pd.DataFrame()
        df['input_tensor'] = 1
        df['output_tensor'] = 2
        df.to_csv(CFG.data_path, index=False)


if __name__ == "__main__":
    understandNet = NeuralNet()
    talkNet = NeuralNet()    
    criterion = nn.MSELoss()
    optimizer_under = optim.Adam(understandNet.parameters(), lr=CFG.lr)
    optimizer_talk = optim.Adam(talkNet.parameters(), lr=CFG.lr)
    
    
    while True:
        cnt = 0
        sentance = torch.randn(100)
        # print(list(talkNet.parameters())[0])
        # print(list(understandNet.parameters())[0])

        with torch.no_grad():
            talkNet.eval()
            talk = talkNet(sentance)
            talk = random_fill_zero(talk)
    
    
        understandNet.eval()
        understand = understandNet(talk)
        understand_loss = criterion(understand, talk)

        if understand_loss < CFG.loss_threshold:
            pass
    
            
        else:
            understandNet.train()
            while True:
                understand = understandNet(sentance)
                # print(list(understandNet.parameters())[0])
                understand_loss = criterion(understand, talk)
                
                optimizer_under.zero_grad()  
                understand_loss.backward()
                
                optimizer_under.step()
                cnt +=1
                with torch.no_grad():
                    understandNet.eval()
                    understand_loss = criterion(understand, talk)
                    if understand_loss < CFG.loss_threshold:
                        break
        write_report(cnt)
        make_weights(understandNet, weights_name='understandNet_weights.pt')
        make_dataframe()
        print(f'cnt:{cnt}')

        
        # print(f'talk:{talk}')
        # print(f'understand:{understand}')