import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from layers import *
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
import pandas as pd
from diffusion_new import *
from loss import *
import time


class HIDE(Module):
    def __init__(self, opt, num_node, adj_all=None, num=None, cat=False):
        super(HIDE, self).__init__()
        # HYPER PARA
        self.opt = opt 
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.layer = int(opt.layer)

        # Item representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.feat_latent_dim = self.dim
        
        img_features_df = pd.read_csv(opt.img_features_path, index_col=0)
        txt_features_df = pd.read_csv(opt.txt_features_path, index_col=0)
        imgWeights = img_features_df.values  
        txtWeights = txt_features_df.values

        img_n_node = imgWeights.shape[0]  
        txt_n_node = txtWeights.shape[0]
        img_emb_size = imgWeights.shape[1]  
        txt_emb_size = txtWeights.shape[1]

        self.image_embedding = nn.Embedding(img_n_node,img_emb_size)
        self.text_embedding = nn.Embedding(txt_n_node,txt_emb_size)
        self.image_embedding.weight.data.copy_(torch.from_numpy(imgWeights))
        self.text_embedding.weight.data.copy_(torch.from_numpy(txtWeights))

        self.img_linear = nn.Linear(img_emb_size, self.dim)
        self.txt_linear = nn.Linear(txt_emb_size, self.dim)
        
        # Position representation
        self.pos_embedding = nn.Embedding(200, self.dim)

        self.local_agg = LocalHyperGATlayer(self.dim, self.layer, self.opt.alpha)
        
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(3 * self.dim, 1))
        self.w_s = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu3 = nn.Linear(self.dim, self.dim, bias=True)
        
        
        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        # main task loss
        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.diffusion = GaussianDiffusion(opt, opt.noise_scale, opt.steps)
        self.InfoNCE = InfoNCE()
        self.beta1 = 0.1
        self.alpha1 = 1

        self.query_common = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, 1, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid()
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)



    def compute_scores(self, hidden, mask, item_embeddings):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        ht = hidden[:, 0, :]
        ht = ht.unsqueeze(-2).repeat(1, len, 1)             # (b, N, dim)
        
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        

        hs = torch.cat([hs, ht], -1).matmul(self.w_s)

        feat = hs * hidden  
        nh = torch.sigmoid(torch.cat([self.glu1(nh), self.glu2(hs), self.glu3(feat)], -1))

        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        
        select = torch.sum(beta * hidden, 1) # 64,200
        b = item_embeddings[1:]  # n_nodes x latent_size 11638,200
        scores = torch.matmul(select, b.transpose(1, 0))

        return scores

    def get_image_embeddings(self):
        image_embeddings = self.image_embedding.weight
        image_embeddings = self.img_linear(image_embeddings)
        return image_embeddings

    def get_text_embeddings(self):
        text_embeddings = self.text_embedding.weight
        text_embeddings = self.txt_linear(text_embeddings)
        return text_embeddings
    
    def diffusion_img(self, inputs, Hs, mask_item, item):
        image_embeddings = self.get_image_embeddings()
        zeros = trans_to_cuda(torch.FloatTensor(1, self.dim).fill_(0))

        image_embeddings = torch.cat([zeros, image_embeddings], 0)
        image_emb = image_embeddings[item] * mask_item.float().unsqueeze(-1)
        session_img = torch.sum(image_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1) # 64,200
        
        # session_img_diff = self.diffusion.p_sample(session_img, self.opt.sampling_steps) + session_img
        session_img_diff = session_img

        return session_img, session_img_diff

    def diffusion_text(self, inputs, Hs, mask_item, item):
        text_embeddings = self.get_text_embeddings()
        zeros = trans_to_cuda(torch.FloatTensor(1, self.dim).fill_(0))

        text_embeddings = torch.cat([zeros, text_embeddings], 0)
        text_emb = text_embeddings[item] * mask_item.float().unsqueeze(-1)
        session_txt = torch.sum(text_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1) # 64,200

        # session_txt_diff = self.diffusion.p_sample(session_txt, self.opt.sampling_steps) + session_txt
        session_txt_diff = session_txt

        return session_txt, session_txt_diff
    
    def align_vt(self, embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)
        
        vt_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()
        
        return vt_loss
    
    def calculate_loss(self, session_item, session_img, session_txt, session_img_diff, session_txt_diff):
            
        img_cross_modal_loss = self.align_vt(session_img, session_img_diff)
        txt_cross_modal_loss = self.align_vt(session_img, session_txt_diff)
        align_vt_loss = img_cross_modal_loss + txt_cross_modal_loss

        align_loss = align_vt_loss 
        
                    
        return align_loss



    def forward(self, inputs, Hs, mask_item, item):
        item_embeddings = self.embedding.weight # 11638, 200

        # multimodal
        image_embeddings = self.get_image_embeddings() # 11638,200
        text_embeddings = self.get_text_embeddings() #11638,200
        item_embeddings = item_embeddings+ 0.1*image_embeddings + 0.15*text_embeddings
        
        # img and txt
        session_img, session_img_diff = self.diffusion_img(inputs, Hs, mask_item, item)
        session_txt, session_txt_diff = self.diffusion_text(inputs, Hs, mask_item, item)

        # item
        zeros = trans_to_cuda(torch.FloatTensor(1, self.dim).fill_(0))
        item_embeddings = torch.cat([zeros, item_embeddings], 0)
        h = item_embeddings[inputs] #64,19,200
        item_emb = item_embeddings[item] * mask_item.float().unsqueeze(-1) #64,19,200
        session_item = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1) # 64,200 

        ## Behaviour
        session_img_diff = torch.multiply(session_item, self.gate_v(session_img_diff))
        session_txt_diff = torch.multiply(session_item, self.gate_t(session_txt_diff))

        session_common = torch.cat([self.query_common(session_img_diff), self.query_common(session_txt_diff)], dim=-1)
        weight_common = F.softmax(session_common, dim=-1)
        session_common = weight_common[:, 0].unsqueeze(dim=1) * session_img_diff + weight_common[:, 1].unsqueeze(dim=1) * session_txt_diff

        sep_image_embeds = session_img_diff - session_common
        sep_text_embeds = session_txt_diff - session_common

        session_img_1 = self.gate_v(session_item)
        session_txt_1 = self.gate_t(session_item)
        sep_session_img = torch.multiply(session_img_1, sep_image_embeds)
        sep_session_txt = torch.multiply(session_txt_1, sep_text_embeds)

        fusion_embeds = (sep_session_img + sep_session_txt + session_common) / 3


        # output
        # session_diff = session_item + 0.1*session_img_diff + 0.15*session_txt_diff + 0.1*session_img + 0.15*session_txt + fusion_embeds
        session_diff = session_item + session_img + session_txt + fusion_embeds
        session_diff_3D = session_diff.unsqueeze(1)
        h_local_diff = self.local_agg(h, Hs, session_diff_3D)


        align_loss = self.calculate_loss(session_item, session_img, session_txt, session_img_diff, session_txt_diff)       
        
        # # diff_loss
        # diff_loss_img = self.diffusion.training_losses(session_img, reweight=True)
        # diff_loss_txt = self.diffusion.training_losses(session_txt, reweight=True)
        # loss_img = diff_loss_img["loss"].mean() * 0.05  
        # loss_txt =  diff_loss_txt["loss"].mean() * 0.05 
        # loss_diff = loss_img + loss_txt

        # total_loss = self.alpha1* align_loss + self.beta1 * loss_diff
        total_loss = 0
        # output
        output = h_local_diff
        return output, item_embeddings, total_loss
 
    

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable.cpu()


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable



def forward(model, data):
    alias_inputs, Hs, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    Hs = trans_to_cuda(Hs).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden, item_embeddings, total_loss = model(items, Hs, mask, inputs)
    # print("hidden shape:", hidden.shape) #64,19,200
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    # print("seq_hidden shape", seq_hidden.shape) # 64,19,200
    return targets, model.compute_scores(seq_hidden, mask, item_embeddings), total_loss


def train_test(model, train_data, test_data, top_K, opt):
    #print('start training: ', datetime.datetime.now())
    start_train_time = time.time()
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    
    for data in tqdm(train_loader): #data[0].shape # 64,19

        model.optimizer.zero_grad()
        targets, scores, new_loss = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss += new_loss

        loss.backward()
        model.optimizer.step()
        total_loss += loss
    model.scheduler.step()

    end_train_time = time.time()
    train_duration = end_train_time - start_train_time

    start_test_time = time.time()
    
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    
    for data in test_loader:
        targets, scores, _ = forward(model, data)
        targets = targets.numpy()
        for K in top_K:
            sub_scores = scores.topk(K)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                metrics['hit%d' % K].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(score == target - 1)[0][0] + 1))
    
    end_test_time = time.time()
    test_duration = end_test_time - start_test_time

    metrics['train_time'] = train_duration
    metrics['test_time'] = test_duration

    return metrics



