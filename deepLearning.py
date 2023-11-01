import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def f(x):
    if type(x)==float:
        X=x
        X_1=x
    else:
        X = float(x.norm())
    
        if x.shape:
            X_1 = float(x[0])
        else:
            X_1 = float(x)
    return X**2-2*X**3+5*X_1**2

def N(mu):
    Normal_mu = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(float(mu)))
    return Normal_mu

def z(x, mu):
    return f(x) + N(mu)

def t(x, mu1):
    return torch.sign(f(x) + N(mu1))

def create_y(x, mu):
    y = []
    for i in x:
        y.append(z(i,mu))
    y = torch.tensor(y)
    return y

def create_yt(x, mu1):
    y = []
    for i in x:
        res = t(i, mu1)
        if res>0.5:
            y.append(1.0)
        else:
            y.append(0.0)
    y = torch.tensor(y).reshape(-1, 1)
    return y

def new_data(mu,n_samples=1000):
    k = 3
    x = torch.randn(n_samples,k)
    y = create_y(x, mu).reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    return x_train, y_train, x_valid, y_valid, X_test, Y_test


input_dim = 3
hidden_dim = 15
output_dim = 1

dataX_by_mu = {}
dataZ_by_mu = {}
dataT_by_mu = defaultdict(lambda: defaultdict(tuple))
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x
    
class Classification3(nn.Module):
    def __init__(self):
        super(Classification3, self).__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x
    


from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
def prep_data(mu=None,mu1=None,isload=True,n_samples=1000):
    if isload and not mu1:
        x_train, x_valid, X_test = dataX_by_mu[mu]
        y_train, y_valid, Y_test = dataZ_by_mu[mu]
    elif not isload and mu1:
        x_train, x_valid, X_test = dataZ_by_mu[mu]
        y_train = create_yt(x_train, mu1)
        y_valid = create_yt(x_valid, mu1)
        Y_test = create_yt(X_test, mu1)
        dataT_by_mu[mu][mu1] = y_train, y_valid, Y_test
        x_train, x_valid, X_test = dataX_by_mu[mu]
    elif mu1:
        x_train, x_valid, X_test = dataX_by_mu[mu]
        y_train, y_valid, Y_test = dataT_by_mu[mu][mu1]
    else:
        x_train, y_train, x_valid, y_valid, X_test, Y_test = new_data(mu,n_samples=n_samples)
        dataX_by_mu[mu] = x_train, x_valid, X_test
        dataZ_by_mu[mu] = y_train, y_valid, Y_test
    
    return (x_train, y_train, x_valid, y_valid, X_test, Y_test)

def results(model=None,mu1=None,testset=None,X_test=None,Y_test=None):
    string=""
    res=[] 
    if not mu1:
        testloss1 = 0
        testloss2 = 0
        with torch.no_grad():
            for batch_x, batch_y in testset:
                yhat = model(batch_x)
                loss = nn.MSELoss()(yhat, batch_y)
                testloss1 += loss.detach().numpy()
        string+="testMSE:"+str(testloss1/len(Y_test))+"\n"
        res.append(testloss1/len(Y_test))
        with torch.no_grad():
                for batch_x, batch_y in testset:
                    yhat = model(batch_x)
                    loss = nn.L1Loss()(yhat, batch_y)
                    testloss2 += loss.detach().numpy()
        string+="testMAE:"+str(testloss2/len(Y_test))
        res.append(testloss2/len(Y_test))
    else:
        with torch.no_grad():
            pred = model(X_test).detach().numpy()
            pred = [1.0 if i>0.5 else 0.0 for i in pred]
            string+="testF1:"+ str(f1_score(pred,Y_test))+"\n"
            string+="testAcc:"+ str(accuracy_score(pred,Y_test))
            res.append(f1_score(pred,Y_test))
            res.append(accuracy_score(pred,Y_test))
    return string, res
    


def train(model=None,data=None, mu1=None, epochs=100, loss_fn=nn.MSELoss(), lr=0.01,adam=False):
    x_train, y_train, x_valid, y_valid, X_test, Y_test = data 
    if adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainset = DataLoader(MyDataset(x_train, y_train), batch_size=64)
    testset = DataLoader(MyDataset(X_test, Y_test), batch_size=64)
    valset = DataLoader(MyDataset(x_valid, y_valid), batch_size=64)

#     print(model)
    # Define the training loop
    trainlosslist = []
    evallosslist = []
    
    for epoch in range(epochs):
        trainloss = 0
        evalloss = 0
        
        for batch_x, batch_y in trainset:
            model.train(True)
            optimizer.zero_grad()

            yhat = model(batch_x)
            loss = loss_fn(yhat, batch_y)
            
            trainloss += loss.detach().numpy()

            loss.backward()
            optimizer.step()

            model.eval()

        with torch.no_grad():
            for batch_x, batch_y in valset:
                yhat = model(batch_x)

                loss = loss_fn(yhat, batch_y)
                evalloss += loss.detach().numpy()

        trainlosslist.append(trainloss/len(y_train))
        evallosslist.append(evalloss/len(y_valid))
    
    string,test = results(model=model, mu1=mu1, testset=testset, X_test=X_test, Y_test=Y_test)
            
    return trainlosslist, evallosslist,string, test

class CustomAdam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Retrieve the hyperparameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['step'] += 1

                # Update biased first moment estimate
                state['m'] = beta1 * state['m'] + (1 - beta1) * p.grad
                # Update biased second raw moment estimate
                state['v'] = beta2 * state['v'] + (1 - beta2) * (p.grad ** 2)

                # Correct bias in first moment estimate and bias in second raw moment estimate
                m_hat = state['m'] / (1 - beta1 ** state['step'])
                v_hat = state['v'] / (1 - beta2 ** state['step'])

                # Update parameters
                p.data = p.data - lr * m_hat / (torch.sqrt(v_hat) + eps)


def train2(model=None,data=None, mu1=None, epochs=100, loss_fn=nn.MSELoss(), lr=0.01,adam=False):
    x_train, y_train, x_valid, y_valid, X_test, Y_test = data 
    optimizer1 = CustomAdam(model.parameters(), lr=lr)
    trainset = DataLoader(MyDataset(x_train, y_train), batch_size=64)
    testset = DataLoader(MyDataset(X_test, Y_test), batch_size=64)
    valset = DataLoader(MyDataset(x_valid, y_valid), batch_size=64)

    # Define the training loop
    trainlosslist = []
    evallosslist = []
    
    for epoch in range(epochs):
    #     print(epoch)
        trainloss = 0
        evalloss = 0
        
        for batch_x, batch_y in trainset:
            model.train(True)
            optimizer1.zero_grad()

            yhat = model(batch_x)
            loss = loss_fn(yhat, batch_y)

            loss.backward()
            optimizer1.step()
            
            optimizer1.zero_grad()
            yhat = model(batch_x)
            loss = loss_fn(yhat, batch_y)
            
            
            trainloss += loss.detach().numpy()
            
              
            loss.backward()
            optimizer1.step()

            model.eval()

        with torch.no_grad():
            for batch_x, batch_y in valset:
                yhat = model(batch_x)

                loss = loss_fn(yhat, batch_y)
                evalloss += loss.detach().numpy()

        trainlosslist.append(trainloss/len(y_train))
        evallosslist.append(evalloss/len(y_valid))
    
    string,test= results(model=model, mu1=mu1, testset=testset, X_test=X_test, Y_test=Y_test)
            
    return trainlosslist, evallosslist, string,test

lis = [(10)**i for i in range(3)]
lis = [0.1]+lis
seed_list = [10,20,30,42,50]

torch.manual_seed(42)
training_error= []
results_mse1=[]
results_mae1=[]
for mu in lis:
    best_results=None
    result_show=None
    final_test=""
    data = prep_data(mu=mu,isload=False)
    for seed in seed_list:
        torch.manual_seed(seed)
        model = NeuralNetwork()
        trainlosslist, evallosslist, string, test = train(model,data=data)
        if not best_results or  best_results>evallosslist[-1]:
            best_results=evallosslist[-1]
            result_show=(trainlosslist, evallosslist)
            final_string=string
            final_test=test
    results_mse1.append(final_test[0])
    results_mae1.append(final_test[1])
    trainlosslist, evallosslist = result_show     
    training_error.append(trainlosslist[-1])

plt.plot(lis,training_error,label=("training error as function of mu"),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("A,mu,MSE_trainingerror.png")
plt.show()

torch.manual_seed(42)
training_error= []
results_mse2=[]
results_mae2=[]
for mu in lis:
    best_results=None
    result_show=None
    final_test=""
    data = prep_data(mu=mu,isload=True)
    for seed in seed_list:
        torch.manual_seed(seed)
        model = NeuralNetwork()
        trainlosslist, evallosslist, string, test= train(model,data=data,loss_fn=nn.L1Loss())
        if not best_results or  best_results>evallosslist[-1]:
            best_results=evallosslist[-1]
            result_show=(trainlosslist, evallosslist)
            final_string=string
            final_test=test
    results_mse2.append(final_test[0])
    results_mae2.append(final_test[1])
    trainlosslist, evallosslist = result_show
    training_error.append(trainlosslist[-1])

plt.plot(lis,results_mse1,label=("MSE function of mu MSE learning"),marker='.')
plt.plot(lis,results_mse2,label=("MSE function of mu MAE learning"),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("A,mu,MSE.png")
plt.show()

plt.plot(lis,results_mae1,label=("MAE function of mu MSE learning"),marker='.')
plt.plot(lis,results_mae2,label=("MAE function of mu MAE learning"),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("A,mu,MAE.png")
plt.show()

torch.manual_seed(42)
training_error= []
training_error_mu100= []
test_f1A=[]
test_accA=[]
for mu in lis:
    for mu1 in lis:
        best_results=None
        result_show=None
        final_test=""
        data = prep_data(mu=mu,mu1=mu1,isload=False)
        for seed in seed_list:
            model = Classification()
            trainlosslist, evallosslist, string, test= train(model,data=data,mu1=mu1,loss_fn=nn.BCELoss(),lr=0.05)
            if not best_results or  best_results>evallosslist[-1]:
                best_results=evallosslist[-1]
                result_show=(trainlosslist, evallosslist)
                final_string=string
                final_test=test
        test_f1A.append(final_test[0])
        test_accA.append(final_test[1])
        trainlosslist, evallosslist = result_show
        if mu == 0.1:
            training_error.append(trainlosslist[-1])
        if mu == 100:
            training_error_mu100.append(trainlosslist[-1])
    
plt.plot(lis,training_error,label=("mu=0.1,training error as function of mu1 "),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("B,mu0.1.png")
plt.show()

plt.plot(lis,training_error_mu100,label=("mu=100,training error as function of mu1 "),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("B,mu100.png")
plt.show()

torch.manual_seed(42)
training_error= []
training_error_mu100= []
test_f1B=[]
test_accB=[]
for mu in lis:
    for mu1 in lis:
        best_results=None
        result_show=None
        final_test=""
        data = prep_data(mu=mu,mu1=mu1,isload=True)
        for seed in seed_list:
            model = Classification()
            trainlosslist, evallosslist, string, test= train(model,data=data,mu1=mu1,loss_fn=nn.HingeEmbeddingLoss(),lr=0.001)
            if not best_results or  best_results>evallosslist[-1]:
                best_results=evallosslist[-1]
                result_show=(trainlosslist, evallosslist)
                final_string=string
                final_test=test
        trainlosslist, evallosslist = result_show  
        if mu == 0.1:
            training_error.append(trainlosslist[-1])
        if mu == 100:
            training_error_mu100.append(trainlosslist[-1])
        test_f1B.append(final_test[0])
        test_accB.append(final_test[1])

plt.plot(lis,training_error,label=("mu=0.1,training error as function of mu1 "),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

plt.plot(lis,training_error_mu100,label=("mu=0.1,training error as function of mu1 "),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

plt.plot(lis,test_f1A[:4],label=("F1 function of mu BCE learning"),marker='.')
plt.plot(lis,test_f1B[:4],label=("F1 function of mu Hinge learning"),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("B,mu,F1,0.1.png")
plt.show()

plt.plot(lis,test_accA[:4],label=("Accuracy function of mu BCE learning"),marker='.')
plt.plot(lis,test_accB[:4],label=("Accuracy function of mu Hinge learning"),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("B,mu,ACC0.1.png")
plt.show()

plt.plot(lis,test_accA[-4:],label=("Accuracy function of mu BCE learning"),marker='.')
plt.plot(lis,test_accB[-4:],label=("Accuracy function of mu Hinge learning"),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("B,mu,ACC100.png")
plt.show()

torch.manual_seed(42)
# for mu in lis:
for mu in lis:
    for mu1 in lis:
        best_results=None
        result_show=None
        final_test=""
        data1 = prep_data(mu=mu,isload=True)
        data2 = prep_data(mu=mu,mu1=mu1,isload=True)
        for seed in seed_list:
            model1 = NeuralNetwork()
            model3 = Classification3()
            trainlosslist, evallosslist,string, test = train(model1,data=data1)

            x_train, y_train, x_valid, y_valid, X_test, Y_test = data1
            with torch.no_grad():
                x_train = model1(x_train)
                x_valid = model1(x_valid)
                X_test = model1(X_test)
            _,y_train,_,y_valid,_,Y_test =data2
            data2 = (x_train, y_train, x_valid, y_valid, X_test, Y_test)
            trainlosslist, evallosslist,string, test = train(model3,data=data2,mu1=mu1,loss_fn=nn.BCELoss(),lr=0.05)
            if not best_results or  best_results>evallosslist[-1]:
                    best_results=evallosslist[-1]
                    result_show=(trainlosslist, evallosslist)
                    final_string=string
        
        if mu==mu1 and mu==0.1:
            print(final_string)
            plt.plot(range(len(trainlosslist)),trainlosslist,label=("mu:"+str(mu)+",mu1:"+str(mu1)+" train"))
            plt.plot(range(len(evallosslist)),evallosslist,label=("mu:"+str(mu)+",mu1:"+str(mu1)+" eval"))
            plt.legend()
            name="D,"+"2net,"+str(mu)+",mu1,"+str(mu1)+".png"
            plt.savefig(name)
            plt.show()
        
torch.manual_seed(42)
one_step_test=[]
for mu in lis:
    for mu1 in lis:
        best_results=None
        result_show=None
        final_test=""
        data = prep_data(mu=mu,mu1=mu1,isload=True)
        for seed in seed_list:
            model = Classification()
            trainlosslist, evallosslist,string, test = train(model,data=data,mu1=mu1,epochs=100,loss_fn=nn.BCELoss(),lr=0.01,adam=True)
            if not best_results or  best_results>evallosslist[-1]:
                best_results=evallosslist[-1]
                result_show=(trainlosslist, evallosslist)
                final_string=string
                final_test=test
        trainlosslist, evallosslist = result_show 
        one_step_test.append(final_test[1])
        
        if mu==mu1 and mu==0.1:
            print(final_string)
            plt.plot(range(len(trainlosslist)),trainlosslist,label=("mu:"+str(mu)+",mu1:"+str(mu1)+" train"))
            plt.plot(range(len(evallosslist)),evallosslist,label=("mu:"+str(mu)+",mu1:"+str(mu1)+" eval"))
            plt.legend()
            name="E,"+"1step,"+str(mu)+",mu1,"+str(mu1)+".png"
            plt.savefig(name)
            plt.show()

torch.manual_seed(42)
two_step_test=[]
for mu in lis:
    for mu1 in lis:
        best_results=None
        result_show=None
        final_test=""
        data = prep_data(mu=mu,mu1=mu1,isload=True)
        for seed in seed_list:
            model = Classification()
            trainlosslist, evallosslist, string, test = train2(model,data=data,mu1=mu1,epochs=100,loss_fn=nn.BCELoss(),lr=0.01,adam=True)
            if not best_results or  best_results>evallosslist[-1]:
                best_results=evallosslist[-1]
                result_show=(trainlosslist, evallosslist)
                final_string=string
                final_test=test
        
        trainlosslist, evallosslist = result_show
        two_step_test.append(final_test[1])
        if mu==mu1 and mu==0.1:
            print(final_string)
            plt.plot(range(len(trainlosslist)),trainlosslist,label=("mu:"+str(mu)+",mu1:"+str(mu1)+" train"))
            plt.plot(range(len(evallosslist)),evallosslist,label=("mu:"+str(mu)+",mu1:"+str(mu1)+" eval"))
            plt.legend()
            name="E,"+"2step,"+str(mu)+",mu1,"+str(mu1)+".png"
            plt.savefig(name)
            plt.show()

plt.plot(lis,one_step_test[:4],label=("Accuracy function of mu1 mu=0.1 One Step Adam"),marker='.')
plt.plot(lis,two_step_test[:4],label=("Accuracy function of mu1 mu=0.1 Two Step Adam"),marker='.')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("E,mu0.1,accAdam.png")
plt.show()

x_pos = torch.tensor(np.linspace(0,3.5,100))
y_out = [f(i) for i in x_pos]
text = "f(x),x>0"
plt.plot(x_pos,y_out)
plt.title(text)
plt.savefig("f(x).png")
plt.show()
        
        

    
    
    

