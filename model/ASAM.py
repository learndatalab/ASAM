import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import networkx as nx
from sklearn.manifold import TSNE
import seaborn as sns
warnings.filterwarnings("error")
initializer = nn.init.xavier_uniform_

def normalize_tensor_batch(tensor_batch, axis=1):
    
    if tensor_batch.shape[2] == 1 or tensor_batch.shape[1] == 1:
        
        return tensor_batch
    mean = tensor_batch.mean(axis, keepdim=True)
    std = tensor_batch.std(axis, keepdim=True) + 1e-6
    normalized_batch = (tensor_batch - mean) / std
    return normalized_batch

class GNN_DA(nn.Module):
    def __init__(self, input_feat, output_feat, indicator):
        super(GNN_DA, self).__init__()
        self.W_gnn = nn.Parameter(initializer(torch.randn(indicator, indicator)))
        # self.W_gnn = nn.Parameter(initializer(torch.randn(36, 36))) 
        self.B_gnn = nn.Parameter((torch.randn(indicator)))
        # self.B_gnn = nn.Parameter((torch.randn(36))) 
        # self.MHA = torch.nn.MultiheadAttention(embed_dim=indicator, num_heads=4, batch_first=True) # 
        self.MHA = torch.nn.MultiheadAttention(embed_dim=indicator, num_heads=4, batch_first=True) # 
        # self.MHA = torch.nn.MultiheadAttention(embed_dim=36, num_heads=4, batch_first=True)
        self.query_embed = nn.Parameter(initializer(torch.randn(1, input_feat, indicator)))
        self.indicator = indicator
        self.output_feat = output_feat
        
    def forward(self, x):
        # query_embed_repeated = self.query_embed.repeat(x.shape[0], 1, 1)
        # a, b = self.MHA(query_embed_repeated, x, x)
        # x2 = torch.bmm(b, x)
        # print(x2[0,:,:5])

        # new_query_embed = self.query_embed[0, torch.randperm(8)]
        # query_embed_repeated = new_query_embed.repeat(x.shape[0], 1, 1)
        # a, b = self.MHA(query_embed_repeated, x,  x)
        # x2 = torch.bmm(b, x)
        # print(x2[0,:,:5])
        # exit(0)

        query_embed_repeated = self.query_embed.repeat(x.shape[0], 1, 1)
        a, b = self.MHA(query_embed_repeated, x, x)
        # x2 = torch.bmm(b, x)
        # print(x2[0,0])

        # # new_query_embed = self.query_embed[0, torch.randperm(8)]
        # new_x = x[:, torch.randperm(8)]
        # # query_embed_repeated = new_query_embed.repeat(x.shape[0], 1, 1)
        # a, b = self.MHA(query_embed_repeated, new_x, new_x)
        # x2 = torch.bmm(b, new_x)
        # print(x2[0,0])
        # exit(0)

        

        # a, b = self.MHA(x, query_embed_repeated,  x)
        # print(b[0])
        # exit(0)
        x = torch.bmm(b, x)
        x = torch.matmul(x, self.W_gnn)
        x += self.B_gnn
        return x


class GNN(nn.Module):
    def __init__(self, input_feat, output_feat, indicator):
        super(GNN, self).__init__()
        self.W_gnn = nn.Parameter(initializer(torch.randn(indicator, indicator)))
        # self.W_gnn = nn.Parameter(initializer(torch.randn(36, 36))) 
        self.B_gnn = nn.Parameter((torch.randn(indicator)))
        # self.B_gnn = nn.Parameter((torch.randn(36))) 
        # self.MHA = torch.nn.MultiheadAttention(embed_dim=indicator, num_heads=4, batch_first=True) # 
        self.MHA = torch.nn.MultiheadAttention(embed_dim=indicator, num_heads=4, batch_first=True) # 
        # self.MHA = torch.nn.MultiheadAttention(embed_dim=36, num_heads=4, batch_first=True)

        self.indicator = indicator
        self.output_feat = output_feat
        
    def forward(self, x):
        a, b = self.MHA(x, x, x)
        x = torch.bmm(b, x)
        x = torch.matmul(x, self.W_gnn)
        x += self.B_gnn
        return x
    
class Spatial_layer(nn.Module):
    def __init__(self, in_dim, out_dim, indicator):
        super(Spatial_layer, self).__init__()
        self.WS_input = torch.nn.Parameter(initializer(torch.randn(out_dim, in_dim, 1, 1)))
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.batch1 = nn.BatchNorm1d(in_dim)
        self.gnn = GNN(in_dim, out_dim, indicator)
        self.gnn2 = GNN(in_dim, out_dim, indicator)
        self.batch2 = nn.BatchNorm1d(in_dim)

        # self.posemb = nn.Parameter(torch.randn(1, out_dim, indicator))
        self.posemb = nn.Parameter(initializer(torch.randn(1, out_dim, indicator)))
        
    def forward(self, x, phase=True):
        if phase:
            x2 = x + self.posemb
        else:
            x2 = x

        x2 = self.gnn(x2)
        
        
        self.batch1(x2)
        x2 = F.relu(x2)

        x2 = self.gnn2(x2)
        x2 = self.batch2(x2)
        x2 = F.relu(x2)
        return x+x2
    
class Spatial_layer_da(nn.Module):
    def __init__(self, in_dim, out_dim, indicator):
        super(Spatial_layer_da, self).__init__()
        self.WS_input = torch.nn.Parameter(initializer(torch.randn(out_dim, in_dim, 1, 1)))
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.batch1 = nn.BatchNorm1d(in_dim)
        self.gnn = GNN_DA(in_dim, out_dim, indicator)
        # self.gnn2 = GNN(in_dim, out_dim, indicator)
        self.batch2 = nn.BatchNorm1d(in_dim)
    def forward(self, x):
        x2 = self.gnn(x)
        self.batch1(x2)
        x2 = F.relu(x2)
        return x+x2


class ABN1d(nn.Module):
  def __init__(self):
    super().__init__()
    self.bn = nn.BatchNorm1d(100)
    # self.bn = nn.BatchNorm1d(36)
    self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
    self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

  def forward(self, x):
    return self.a * x + self.b * self.bn(x)


# model for phase 1
class ARMBANDGNN_modified_rnn_raw(nn.Module):
    def __init__(self, electrode_num, channels, num_classes, input_feature_dim):
        super(ARMBANDGNN_modified_rnn_raw, self).__init__()
        
        self.Spat1 = Spatial_layer(electrode_num, electrode_num, input_feature_dim)#(second, second, indicator)
        # self.batch2 = ABN1d()
        self.batch2 = nn.BatchNorm1d(input_feature_dim)#(third)
        # self.batch2 = nn.BatchNorm1d(100)#(third)

        # self.lstm2 = nn.LSTM(input_size=electrode_num, hidden_size=512, #proj_size=512, 
        #                     num_layers=2, batch_first=True, dropout=0.5)
        # self.linear = nn.Linear(512, num_classes)
        

        self.lstm2 = nn.LSTM(input_size=electrode_num, hidden_size=512, #proj_size=512, 
                            num_layers=2, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(512, num_classes)
    def forward(self, x, phase=True):
        #input shape B, C, T   [b, 8, 100]
        
        if(x.shape[2] != 100 ) and x.shape[2] != 36: # for data formation for different dataset
            x = x.transpose(1, 2)

        if x.shape[1] == 1:
            x = x.transpose(1, 2)
        
        x = self.Spat1(x, phase)   #bct

        ##
        s1_pos = self.Spat1(torch.zeros(x.shape, device=x.device), phase).clone()
        ##
        s1 = x.clone()
        
        # h_1 = Variable(torch.zeros(2, x.size(0), 16)).to(x.device) #hidden state
        # c_1 = Variable(torch.zeros(2, x.size(0), 16)).to(x.device)

        h_1 = Variable(torch.zeros(2, x.size(0), 512)).to(x.device) #hidden state
        c_1 = Variable(torch.zeros(2, x.size(0), 512)).to(x.device)

        x = torch.transpose(x, 1, 2)
        x, (hn, cn) = self.lstm2(x, (h_1, c_1))
        x = self.batch2(x).squeeze(dim=0) #btc
        t2 = x

        x = self.linear(x[:, -1, :])
        pred = F.log_softmax(x, dim=1)
        
        z_s1 = normalize_tensor_batch(s1)
        
        z_t2 = normalize_tensor_batch(t2)

        z_s1_pos = normalize_tensor_batch(s1_pos)
        
        

        return z_s1, z_t2, pred, z_s1_pos
    
# model for da phase

count = 0
class DA_gnn_invariance_ver2(nn.Module):
    def __init__(self, basic_model, electrode_num, channel):
        super(DA_gnn_invariance_ver2, self).__init__()
        self.basic_model = basic_model
        self.MLP1 = nn.Linear(channel, 300)
        self.MLP2 = nn.Linear(300,100)
        self.MLP3 = nn.Linear(100, 100)
        self.drop1 = nn.Dropout2d(p=0.1)
        self.drop2 = nn.Dropout2d(p=0.01)
        self.Spat = Spatial_layer_da(electrode_num, electrode_num, channel)
    def plot(self, x, count, new_x = None):
        if(x.shape[2] != 100 ) and x.shape[2] != 36 and x.shape[2] != 1:
            x = x.transpose(1, 2)

        if new_x is None:
            new_x = x[:, torch.randperm(x.shape[1])]
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.decomposition import PCA
        bs, c, t = x.size()
        # Concatenate x and new_x
        combined_data = torch.cat([x.reshape(x.shape[0], -1), new_x.reshape(new_x.shape[0], -1)])
        # combined_labels = torch.cat([all_source_labels, all_target_labels])
        combined_domain = torch.cat([torch.zeros(x.size(0)), torch.ones(new_x.size(0))])

        # Apply t-SNE to the combined data
        tsne = TSNE(n_components=2, random_state=42)
        combined_tsne = tsne.fit_transform(combined_data.cpu().detach().numpy())

        # Create a DataFrame for visualization
        combined_df = pd.DataFrame(combined_tsne, columns=['Dim1', 'Dim2'])
        # combined_df['Label'] = combined_labels.cpu().numpy()
        combined_df['Domain'] = combined_domain.cpu().numpy()

        # Plot the t-SNE results with KDE
        plt.figure(figsize=(8, 8))
        # sns.scatterplot(x='Dim1', y='Dim2', hue='Domain', style='Label', data=combined_df, palette='viridis')
        # sns.scatterplot(x='Dim1', y='Dim2', hue='Domain', data=combined_df, palette='viridis', alpha=0.8)
        sns.kdeplot(x='Dim1', y='Dim2', data=combined_df, fill=True, palette='viridis', hue='Domain', alpha=0.65)
        # plt.title(f'Combined Source and Target Data t-SNE_{count}')
        plt.xticks([])
        plt.xlabel('')
        plt.yticks([])
        plt.ylabel('')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        # plt.savefig(f'tsne_original_{count}.png')
        plt.savefig(f'tsne_original_{count}_har.png')
        # # exit(0)
        
        
        x2 = self.Spat(x) - 0.9*x#torch.mean(x, dim=1, keepdim=True).repeat(1, 8, 1)
        new_x2 = self.Spat(new_x) - 0.9*new_x#torch.mean(new_x, dim=1, keepdim=True).repeat(1, 8, 1)
        # Concatenate x and new_x
        combined_data = torch.cat([x2.reshape(x2.shape[0], -1), new_x2.reshape(new_x2.shape[0], -1)])
        # combined_labels = torch.cat([all_source_labels, all_target_labels])
        combined_domain = torch.cat([torch.zeros(x2.size(0)), torch.ones(new_x2.size(0))])

        # Apply t-SNE to the combined data
        tsne = TSNE(n_components=2, random_state=42)
        combined_tsne = tsne.fit_transform(combined_data.cpu().detach().numpy())

        # Create a DataFrame for visualization
        combined_df = pd.DataFrame(combined_tsne, columns=['Dim1', 'Dim2'])
        # combined_df['Label'] = combined_labels.cpu().numpy()
        combined_df['Domain'] = combined_domain.cpu().numpy()

        # Plot the t-SNE results with KDE
        plt.figure(figsize=(8, 8))
        # sns.scatterplot(x='Dim1', y='Dim2', hue='Domain', style='Label', data=combined_df, palette='viridis')
        # sns.scatterplot(x='Dim1', y='Dim2', hue='Domain', data=combined_df, palette='viridis', alpha=0.8)
        sns.kdeplot(x='Dim1', y='Dim2', data=combined_df, fill=True, palette='viridis', hue='Domain', alpha=0.65)
        # plt.title(f'Combined Source and Target Data t-SNE_{count}')
        plt.xticks([])
        plt.xlabel('')
        plt.yticks([])
        plt.ylabel('')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        # plt.savefig(f'tsne_after_transform_{count}_minux_x_0.9.png')
        plt.savefig(f'tsne_after_transform_{count}_minux_x_har_0.9.png')
    
    def forward(self, x):
        if(x.shape[2] != 100 ) and x.shape[2] != 36 and x.shape[2] != 1:
            x = x.transpose(1, 2)
        
        
        x2 = self.Spat(x)
        
        inv_1 = normalize_tensor_batch(x2)
        
        

        z_s1, z_t2, pred, z_s1_pos = self.basic_model(x, phase=False)
        inv_2 = z_s1
        
        # pos_s1, pos_t2, _ = self.basic_model(0)

        z_s1, z_t2, pred, z_s2_pos = self.basic_model(x2, phase=False)
        
        return inv_2, z_s1, pred, z_s1_pos
        return inv_1, inv_2, pred, z_s1_pos



