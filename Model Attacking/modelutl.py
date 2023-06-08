import torch

class DLModel(object):

    def __init__(self, model, opt):
        self.model = model
        self.opt = opt
    
    def train(self):
        self.model.train()
        
    def zero_grad(self):
        self.opt.zero_grad()

    def forward(self, x):
        return self.model(x)
        
    def backward(self):
        self.opt.step()
        
    def eval(self):
        self.model.eval()

    def inference(self, x):
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(x)
            
        return output

