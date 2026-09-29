import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k):
        super().__init__(); self.k=k
        self.conv=nn.Sequential(nn.Conv1d(k,64,1),nn.BatchNorm1d(64),nn.ReLU(),nn.Conv1d(64,128,1),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,1024,1),nn.BatchNorm1d(1024),nn.ReLU())
        self.fc1=nn.Linear(1024,512); self.bn1=nn.BatchNorm1d(512); self.fc2=nn.Linear(512,256); self.bn2=nn.BatchNorm1d(256); self.fc3=nn.Linear(256,k*k)
        nn.init.zeros_(self.fc3.weight); nn.init.zeros_(self.fc3.bias)
    def forward(self,x):
        b=x.size(0); x=torch.max(self.conv(x),2).values; x=F.relu(self.bn1(self.fc1(x))); x=F.relu(self.bn2(self.fc2(x))); x=self.fc3(x)
        return x.view(b,self.k,self.k)+torch.eye(self.k,device=x.device,dtype=x.dtype).unsqueeze(0)

class PointNet(nn.Module):
    def __init__(self,num_classes=40):
        super().__init__(); self.input_tnet=TNet(3); self.feature_tnet=TNet(64)
        self.c1=nn.Conv1d(3,64,1); self.b1=nn.BatchNorm1d(64); self.c2=nn.Conv1d(64,64,1); self.b2=nn.BatchNorm1d(64); self.c3=nn.Conv1d(64,64,1); self.b3=nn.BatchNorm1d(64); self.c4=nn.Conv1d(64,128,1); self.b4=nn.BatchNorm1d(128); self.c5=nn.Conv1d(128,1024,1); self.b5=nn.BatchNorm1d(1024)
        self.fc1=nn.Linear(1024,512); self.bn6=nn.BatchNorm1d(512); self.fc2=nn.Linear(512,256); self.bn7=nn.BatchNorm1d(256); self.fc3=nn.Linear(256,num_classes); self.drop=nn.Dropout(0.3)
    def forward(self,points):
        ti=self.input_tnet(points); x=torch.bmm(points.transpose(1,2),ti).transpose(1,2); x=F.relu(self.b1(self.c1(x))); x=F.relu(self.b2(self.c2(x)))
        tf=self.feature_tnet(x); x=torch.bmm(x.transpose(1,2),tf).transpose(1,2); x=F.relu(self.b3(self.c3(x))); x=F.relu(self.b4(self.c4(x))); x=self.b5(self.c5(x)); x=torch.max(x,2).values
        x=F.relu(self.bn6(self.fc1(x))); x=F.relu(self.bn7(self.drop(self.fc2(x)))); return self.fc3(x),ti,tf

def feature_transform_regularizer(t):
    k=t.size(1); i=torch.eye(k,device=t.device,dtype=t.dtype).unsqueeze(0); return torch.linalg.matrix_norm(torch.bmm(t,t.transpose(1,2))-i,ord='fro',dim=(1,2)).mean()

if __name__=='__main__':
    m=PointNet().eval(); y,ti,tf=m(torch.randn(2,3,1024)); print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}'); print(y.shape,ti.shape,tf.shape)
