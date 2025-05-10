import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


##########################
# Examples of EDO        #
##########################

class EDO:
    def __init__(self, f, d_yf, t0, y0):
        self.f = f
        self.d_yf = d_yf
        self.t0 = t0
        self.y0 = y0


EDO_Square_limit_cycle = lambda δ, t0, y0 : EDO(lambda y,t : np.array([(-δ*y[0]+y[1])*(y[0]**2 - 1), -(δ*y[1] + y[0])*(y[1]**2 - 1)]), \
                         d_yf = lambda y,t, h: np.array([(-δ*h[0] + h[1])*(y[0]**2 - 1) + (-δ*y[0]+y[1])*2*h[0]*y[0],-(δ*h[1] + h[0])*(y[1]**2 - 1) -(δ*y[1] + y[0])*2*h[1]*y[1]]),\
                         t0 = t0, y0 = y0)
EDO_Example_linear = lambda a, t0,y0 : EDO(lambda y,t : a*y*(np.sin(t))**2 , lambda y,t, h : a*h*(np.sin(t))**2, t0 = t0, y0 = y0)
EDO_Example_non_linear = lambda a, t0,y0 : EDO(lambda y,t : a*(y**2)*(np.sin(t))**2 , lambda y,t, h : 2*a*y*h*(np.sin(t))**2, t0 = t0, y0 = y0)
EDO_CosSin = lambda a, b, t0,y0: EDO(lambda y,t : np.array([-a*y[1], b*y[0]]), lambda y,t,h: np.array([-a*h[1], b*h[0]]) ,t0,y0)
EDO_Lorentz_system = lambda σ, ρ , β, t0, y0 : EDO(lambda y,t : np.array([σ*(y[1] - y[0]), y[0]*(ρ - y[2]) - y[1], y[0]*y[1] - β*y[2]]), lambda y,t,h: np.array([σ*(h[1] - h[0]), y[0]*(ρ - h[2]) + h[0]*(ρ - y[2]) - h[1], y[0]*h[1] + h[0]*y[1] - β*h[2]]), t0, y0) 

###########################
#Discrete solvers EDO     #
###########################

def Runge_Kutta_method(edo, list_t ,s=4):
    if s == 4: 
        a = np.array([[0,0,0,0],[1/2,0,0,0],[0,1/2,0,0],[0,0,1,0]])
        c = np.sum(a, axis = 1)
        b = np.array([1/6, 1/3, 1/3, 1/6])
    d = len(edo.y0)
    y,old_t  = edo.y0, edo.t0
    list_y = [y]
    
    for t in list_t[:-1]:
        h = t - old_t
        k = np.zeros((s,d))
        for i in range(s):
            k[i] = edo.f(y + h*np.sum([a[i,j]*k[j] for j in range(s)], axis = 0 ),t + c[i]*h )
        y = y + h * np.sum([k[i] * b[i] for i in range(s)], axis = 0 )
        list_y.append(y)
        old_t = t
    return list_y

Odeint_method = lambda edo, list_t : odeint(edo.f, edo.y0, list_t)

def give_diff_edo(edo,list_t, method):
    list_y = method(edo,list_t)
    N =len(list_t)
    edo_diff = lambda x : EDO(lambda y,t : edo.d_yf(list_y[min(np.searchsorted(list_t, t, side='right'), N-1)],t,y),0, edo.t0, x)
    return list_y, edo_diff
    
def solve_diff_edo(edo,list_t,method):
    list_y, edo_diff = give_diff_edo(edo,list_t, method)
    diff_y = lambda x : method(edo_diff(x), list_t)
    return list_y, diff_y

###########################
#Visualize functions      #
###########################

def tracer_courbe(liste_points, titre='Courbe', label_x='x', label_y='y'):
    x = [point[0] for point in liste_points]
    y = [point[1] for point in liste_points]

    plt.plot(x, y)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(titre)
    plt.grid(True)
    plt.show()
    
def afficher_graphe(t_values, y_values):
    # Affichage du graphe
    plt.plot(t_values, y_values, linestyle='-', color='b', label='y')
    plt.title("y")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
def display_graph(list_t, list_x, title = "Graph"):
    L = len(list_t)
    d = len(list_x[0])

    if d == 1:
        # Create a 2D graph for d = 1
        plt.plot(list_t, [x[0] for x in list_x])
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(title)
        plt.show()

    elif d == 2:
        # Create a 2D graph for d = 2
        plt.plot([x[0] for x in list_x], [x[1] for x in list_x])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.show()

    elif d == 3:
        # Create a 3D graph for d = 3
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([x[0] for x in list_x], [x[1] for x in list_x], [x[2] for x in list_x])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(title)
        plt.show()

    else:
        print("The dimension d must be 1, 2, or 3.")
###########################
#Deep Learning            #
###########################

def genererate_normalized_vectors(n, d):
    vectors = np.random.normal(size=(n, d))
    normes = np.linalg.norm(vectors, axis=1)
    normalized_vectors = vectors / normes[:, np.newaxis]
    return normalized_vectors
def generate_vectors_with_random_norm(n, d, a, b):
    vectors = np.random.normal(size=(n, d))
    normes = np.linalg.norm(vectors, axis=1)
    normalized_vectors = vectors / normes[:, np.newaxis]
    random_normes = np.random.uniform(a, b, size=n)
    vectors_with_random_norm = normalized_vectors * random_normes[:, np.newaxis]
    return vectors_with_random_norm
    
def create_dataset(edo,method,n, m, init_interval,time_init_interval,time_delta_interval):
    d = len(edo.y0)
    print(d)
    if d == 1:
        list_y0 = np.random.uniform(init_interval[0], init_interval[1], m)
    else:
        list_y0 = generate_vectors_with_random_norm(m, d, init_interval[0], init_interval[1])
    list_h = genererate_normalized_vectors(m,d)
    T, X, Y = torch.zeros((m,n)),torch.zeros((m,2,d)), torch.zeros((m,2,n,d))
    for index ,(y0,h) in enumerate(zip(list_y0, list_h)):
        if d == 1:
            y0 = [y0]
        t0 = np.random.uniform(time_init_interval[0], time_init_interval[1])
        time_delta = np.random.uniform(time_delta_interval[0], time_delta_interval[1])
        n_list_t = np.linspace(t0, t0+time_delta,n) 
        print("y0:", y0,"t0:",t0, "h: ", h, "T:", time_delta, "delta_t: ", time_delta/n)
        edo.y0 = y0
        edo.t0 = t0
        list_y, edo_diff = give_diff_edo(edo,n_list_t, method)
        list_diff_y = method(edo_diff(h), n_list_t)
        Y[index] = torch.tensor([list_y,  list_diff_y])
        X[index] = torch.tensor([y0,h])
        T[index] = torch.tensor(n_list_t)
    return T, X,Y


class TrainDataset(Dataset):
    def __init__(self, T, X, Y):
        self.X = X
        self.Y = Y
        self.T = T
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, ind):
        x = self.X[ind]
        y = self.Y[ind]
        t = self.T[ind]
        return t, x, y

class MLP(nn.Module):
    def __init__(self,in_features : int,out_features: int,hidden_features: int,num_hidden_layers: int) -> None:
        super().__init__()

        
        self.linear_in = nn.Linear(in_features,hidden_features)
        self.linear_out = nn.Linear(hidden_features,out_features)
        
        self.activation = nn.GELU() # nn.SiLU()
        self.layers = nn.ModuleList([self.linear_in] + [nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers)  ])
        
         
    def forward(self,x):
        for layer in self.layers:
            x = self.activation(layer(x))
    
        return self.linear_out(x)


class DeepONet(nn.Module):
    def __init__(self,latent_features,out_features,branch,trunk) -> None:
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.fc = nn.Linear(latent_features,out_features,bias = False)
        

    def forward(self,t,v0):
        #v0 = (t0, u0)
        return self.fc(self.trunk(t)*self.branch(v0))
        
def train_neural_network(model, num_epochs, Loss, train_loader,optimizer):
    loss_history = []
    for n in range(num_epochs):
        mean_loss = 0
        with tqdm(train_loader, desc=f'Epoch {n+1}') as pbar:
            for batch_num, input_data in enumerate(pbar):
                t, xh, y = input_data
                xh = xh.to(torch.double)
                t = t.to(torch.double).unsqueeze(2)
                x, h = torch.unsqueeze(xh[:,0],1), torch.unsqueeze(xh[:,1],1)
                func = lambda x : model(t,torch.cat((t[:,0].unsqueeze(2),x), dim = 2))
                y_pred, dy_pred = torch.func.jvp(func, (x,) ,(h,))
                loss = Loss(y_pred, y[:,0],dy_pred,y[:,1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        epoch_loss = mean_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f'Epoch {n+1}, Average Loss: {epoch_loss:.4f}')
    
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss en fonction des époques')
    plt.show()
    return loss_history

#########################################
#Tests                                  #
#########################################

def fun_N(Onet, t,t0, y0, h):
    t = torch.tensor(t).to(torch.double).unsqueeze(0).unsqueeze(2)
    func = lambda x : Onet(t,torch.cat((t[:,0].unsqueeze(2),x), dim = 2))
    y0, h = torch.tensor(y0, dtype = torch.double), torch.tensor(h, dtype = torch.double)
    y0, h = y0.unsqueeze(0).unsqueeze(1), h.unsqueeze(0).unsqueeze(1)
    y_pred, dy_pred = torch.func.jvp(func, (y0,) ,(h,))
    y_pred, dy_pred = y_pred[0].detach().numpy(), dy_pred[0].detach().numpy()
    return y_pred, dy_pred
    
def test_Onet(edo, Onet, list_t, y0, h):
    edo.y0 = y0
    
    y_pred, dy_pred = fun_N(Onet, list_t,list_t[0], y0, h)
    list_y, diff_y = solve_diff_edo(edo,list_t,Odeint_method)
    list_diff_y = diff_y(h)
    
    display_graph(list_t,(y_pred - list_y), "Diff pred/real")
    display_graph(list_t,y_pred, "Sol pred")
    display_graph(list_t,list_y, "Sol real")
    display_graph(list_t,(dy_pred - list_diff_y),"Diff pred/real edo diff")
    display_graph(list_t,dy_pred, "Sol pred edo diff")
    display_graph(list_t,list_diff_y, "Sol real edo diff")
    return y_pred, list_y, dy_pred, list_diff_y

###############################################
# Parareal                                    #
###############################################

def parareal_NN(y0, G_fine, NN, list_t_interface, N, num_iter, method = 1):
    k = len(list_t_interface)
    list_t_fine = np.linspace(list_t_interface[0],list_t_interface[-1], (k-1)*N)
    iterates = []
    if method == 1:
        # Init 1
        list_y_init = np.array([y0])
        for i in range(k-1):
            subsection = list_t_fine[i*N: (i+1)*N]
            list_y_pred, _  = NN(subsection, list_y_init[-1], list_y_init[-1])
            list_y_init  = np.concatenate((list_y_init, list_y_pred))
        iterates.append(list_y_init[1:])
        display_graph(list_t_fine, iterates[-1], "Initialization, with NN step by step")
    if method == 2:
        # Init 2
        list_t_coarse = np.linspace(list_t_interface[0],list_t_interface[-1], N)
        list_y_init, _  = NN(list_t_coarse,y0,y0)
        list_y_init_fine = np.array([y0])
        for t in list_t_fine:
            z = list_y_init[min(np.searchsorted(list_t_coarse, t), len(list_t_coarse) - 1)]
            list_y_init_fine = np.concatenate((list_y_init_fine,np.array([z])))
        iterates.append(list_y_init_fine[1:])
        display_graph(list_t_fine, iterates[-1],"Initialization, with NN")
    display_graph(list_t_fine, G_fine(list_t_fine, y0),"Real solution")
    
    for j in range(num_iter):
        for i in range(k-1):
            if i == 0:
                list_y  = np.array(G_fine(list_t_fine[:N], y0))
            else:
                subsection = list_t_fine[i*N: (i+1)*N]
                list_y_fine = G_fine(subsection, iterates[-1][i*N-1])
                _, list_dy_pred  = NN(subsection, iterates[-1][i*N-1], list_y[i*N-1] - iterates[-1][i*N-1])
                list_y_subsection = np.array(list_y_fine) + np.array(list_dy_pred)
                list_y  = np.concatenate((list_y, list_y_subsection))
        iterates.append(list_y)
        display_graph(list_t_fine, iterates[-1], f"Iterate number {j}")
    return iterates
    
def parareal_class(y0, G_fine, G_coarse, list_t_interface, N_fine,N_coarse, num_iter):
    k = len(list_t_interface)
    list_t_fine = np.linspace(list_t_interface[0],list_t_interface[-1], (k-1)*N_fine)
    list_t_coarse = np.linspace(list_t_interface[0],list_t_interface[-1], (k-1)*N_coarse)
    iterates = []
    
    # Initialisation        
    list_y_init = G_coarse(list_t_coarse, y0)
    list_y_init_fine = np.array([y0])
    for t in list_t_fine:
        z = list_y_init[min(np.searchsorted(list_t_coarse, t), len(list_t_coarse) - 1)]
        list_y_init_fine = np.concatenate((list_y_init_fine,np.array([z])))
    iterates.append(list_y_init_fine[1:])
    display_graph(list_t_fine, iterates[-1])
    display_graph(list_t_fine, G_fine(list_t_fine, y0))
    
    for j in range(num_iter):
        for i in range(k-1):
            if i == 0:
                list_y  = np.array(G_fine(list_t_fine[:N_fine], y0))
            else:
                subsection_coarse = list_t_coarse[i*N_coarse:(i+1)*N_coarse]
                subsection_fine = list_t_fine[i*N_fine: (i+1)*N_fine]
                list_y_fine = G_fine(subsection_fine, iterates[-1][i*N_fine-1])
                list_y_coarse_1 = G_coarse(subsection_coarse, list_y[-1])
                list_y_coarse_2 = G_coarse(subsection_coarse, iterates[-1][i*N_fine-1])
                list_y_subsection = np.array(list_y_fine) + list_y_coarse_1[-1] - list_y_coarse_2[-1]
                list_y  = np.concatenate((list_y, list_y_subsection))
        iterates.append(list_y)
        display_graph(list_t_fine, iterates[-1])
    return iterates