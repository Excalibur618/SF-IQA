import torch
import torch.nn as nn
from .gather import GatherLayer


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size

    def forward(self, z_i, z_j, z_n, dist_labels):

            N=2*z_i.shape[0]*self.world_size

            z=torch.cat((z_i,z_j),dim=0)
            dist_labels=torch.cat((dist_labels,dist_labels),dim=0)

            if self.world_size > 1:
                z = torch.cat(GatherLayer.apply(z), dim=0)
                dist_labels = torch.cat(GatherLayer.apply(dist_labels), dim=0)

            z = nn.functional.normalize(z, p=2, dim=1)
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
            #sim=torch.mm(z,z.T)/self.temperature
            #print(sim)
        
            #sim_ij = torch.diag(sim, self.batch_size)
            #sim_ji = torch.diag(sim, -self.batch_size)
            #print(sim_ij.shape)
            #positives = torch.cat([sim_ij, sim_ji], dim=0)
            #print(positives.shape)
            
            dist_labels = dist_labels.cpu()

            positive_mask = torch.mm(dist_labels.to_sparse(), dist_labels.T)
            positive_mask = positive_mask.fill_diagonal_(0).to(sim.device)
            zero_diag = torch.ones((N,N)).fill_diagonal_(0).to(sim.device)
            positive_sum = torch.sum(positive_mask, dim=1)

            nominator =torch.sum(sim*positive_mask,dim=1)
            y=torch.exp(sim / self.temperature)
            denominator = torch.sum(torch.exp(sim)*zero_diag,dim=1)

            #print(nominator)
            #print(denominator)
            loss_partial =nominator / denominator
            loss = torch.sum(loss_partial) / (2 * self.batch_size)
            return loss

    '''

    def forward(self, z_i, z_j, dist_labels):
        
        N = 2 * z_i.shape[0] * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        dist_labels = torch.cat((dist_labels, dist_labels),dim=0)
        
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)
            dist_labels = torch.cat(GatherLayer.apply(dist_labels), dim=0)
        
        # calculate similarity and divide by temperature parameter
        z = nn.functional.normalize(z, p=2, dim=1)
        print(z.shape)
        print(z_i.shape)
        sim = torch.mm(z, z.T) / self.temperature
        dist_labels = dist_labels.cpu()
        
        positive_mask = torch.mm(dist_labels.to_sparse(), dist_labels.T)
        positive_mask = positive_mask.fill_diagonal_(0).to(sim.device)
        zero_diag = torch.ones((N, N)).fill_diagonal_(0).to(sim.device)
        
        # calculate normalized cross entropy value
        positive_sum = torch.sum(positive_mask, dim=1)
        denominator = torch.sum(torch.exp(sim)*zero_diag,dim=1)
        y=torch.sum(sim * positive_mask, dim=1)
        print(y.shape)
        print(positive_sum.shape)
        loss = torch.mean(torch.log(denominator) - \
                          (torch.sum(sim * positive_mask, dim=1)/positive_sum))
        
        return loss
        '''
        
            