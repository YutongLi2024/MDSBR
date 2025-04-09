import torch
from torch import nn
import torch.nn.functional as F
from pytorch_metric_learning import miners, distances, losses, reducers


class MatchLoss(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.T = temperature

    def compute_match_loss(self, feature_left, feature_right, device):
        n = len(feature_left)
        similarity = F.cosine_similarity(feature_left.unsqueeze(1), feature_right.unsqueeze(0), dim=2).to(device)
        similarity = torch.exp(similarity / self.T)

        mask_pos = torch.eye(n, n, device=device, dtype=bool)
        sim_pos = torch.masked_select(similarity, mask_pos)

        sim_total_row = torch.sum(similarity, dim=0)
        loss_row = torch.div(sim_pos, sim_total_row)
        loss_row = -torch.log(loss_row)

        sim_total_col = torch.sum(similarity, dim=1)
        loss_col = torch.div(sim_pos, sim_total_col)
        loss_col = -torch.log(loss_col)

        loss = loss_row + loss_col
        loss = torch.sum(loss) / (2 * n)

        return loss

    def forward(self, session_item, session_img, session_txt):
        device = session_item.device
        loss_img_txt = self.compute_match_loss(session_img, session_txt, device)
        loss_txt_img = self.compute_match_loss(session_txt, session_img, device)

        total_loss = (loss_txt_img+ loss_img_txt) / 2

        return total_loss


class MetricLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()

        self.miner = miners.MultiSimilarityMiner()
        self.distances = distances.CosineSimilarity()
        self.reducer = reducers.MeanReducer()

        self.metric_trip_loss = losses.TripletMarginLoss(
            distance=self.distances,
            reducer=self.reducer,
        )

        self.metric_cont_loss = losses.ContrastiveLoss(
            distance=self.distances,
            pos_margin=1,
            neg_margin=0
        )

        self.cont_loss = ContrastiveLoss(temperature=temperature)
        # self.cont_loss = InfoNCELoss(temperature=temperature)

    def forward(self, AMP_emb, nonAMP_emb, loss_type="cont"):
        assert loss_type in {"cont", "metric_cont", "metric_trip"}, print("metric_loss_type error")

        emb = torch.concat((AMP_emb, nonAMP_emb), dim=0)

        AMP_label = torch.ones(len(AMP_emb), device=AMP_emb.device)
        nonAMP_label = torch.zeros(len(nonAMP_emb), device=nonAMP_emb.device)
        label = torch.concat((AMP_label, nonAMP_label))

        if loss_type == "cont":
            loss = self.cont_loss(emb, label)
        elif loss_type == "metric_cont":
            loss = self.metric_cont_loss(emb, label)
        else:
            hard_pairs = self.miner(emb, label)
            # loss = self.trip_loss(emb, label, hard_pairs)
            loss = self.metric_trip_loss(emb, label, hard_pairs)

        return loss
      
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        n = labels.shape[0]
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

        mask_pos = torch.ones_like(similarity_matrix, device=features.device) * (
            labels.expand(n, n).eq(labels.expand(n, n).t()))
        mask_neg = torch.ones_like(mask_pos, device=features.device) - mask_pos

        similarity_matrix = torch.exp(similarity_matrix / self.T)

        mask_diag = (torch.ones(n, n) - torch.eye(n, n)).to(features.device)
        similarity_matrix = similarity_matrix * mask_diag

        sim_pos = mask_pos * similarity_matrix
        sim_neg = similarity_matrix - sim_pos
        sim_neg = torch.sum(sim_neg, dim=1).repeat(n, 1).T
        sim_total = sim_pos + sim_neg

        loss = torch.div(sim_pos, sim_total)
        loss = mask_neg + loss + torch.eye(n, n, device=features.device)
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)

        return loss


class InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features1, features2, temperature):
        if temperature is None:
            temperature = self.temperature
        
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        similarity_matrix = torch.matmul(features1, features2.T)
        
        positives = torch.diag(similarity_matrix)
        
        similarity_matrix = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0]
        
        exp_sim = torch.exp(similarity_matrix / temperature)

        pos_exp_sim = torch.exp(positives / temperature)
        
        denominator = torch.sum(exp_sim, dim=1) - pos_exp_sim
        
        log_prob = torch.log(pos_exp_sim / denominator)
        
        loss = -log_prob.mean()
        
        return loss