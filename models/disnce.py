import torch
from torch import nn

# Disentangle NCE loss in MoCo manner
class DisNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt


    # feat_B for the background, feat_R for the rain
    # shape: (num_patches * batch_size, feature length)
    def forward(self, featB, featR):
        batch_size = self.opt.batch_size
        num_patches = featB.shape[0]/batch_size
        if featR.shape[0] != num_patches*batch_size:
            raise ValueError('num_patches of rain and background are not equal')

        # making labels
        labels = torch.cat([torch.ones(num_patches,1), torch.zeros(num_patches, 1)], dim=0)

        loss_dis_total = 0
        # obtain each background and the rain layer to calculate the disentangle loss
        for i in range(0, batch_size):
            cur_featB = featB[i*num_patches:(i+1)*num_patches,:]
            cur_featR = featR[i*num_patches:(i+1)*num_patches,:]
            cur_disloss = self.cal_each_disloss(cur_featB, cur_featR, labels)
            loss_dis_total += cur_disloss

            
    # cur_featB: [num_patches, feature length]
    # labels: [num_patches*2, 1]
    def cal_each_disloss(cur_featB, cur_featR, labels):
        featFusion = torch.cat([cur_featB, cur_featR], dim=0)
        mask = torch.eq(labels, labels.t()).float()

        # contrast_count: number of all the rain and background patches
        contrast_feature = featFusion
        contrast_count = features.shape[1]
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits of all the elements
        # Denoting: zi: one sample, zl: all the other samples, zp: positives to zi, za: negatives to zi
        # anchor_dot_contrast = zi * zl
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.opt.nce_T)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask, repeat the masks to match the n_views of positives
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.ones_like(mask).scatter_(1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0)
        mask = mask * logits_mask

        # compute log_prob
        # exp_logits: exp(zi * zl)
        exp_logits = torch.exp(logits) * logits_mask
        # the meaning of sum(1): sum the cosine similarity of one sample and all the other samples
        # log_prob: (zi*zl) - log(sum(exp(zi,zl))) = log[exp(zi*zl) / sum(exp(zi * zl)) ]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))


        # compute mean of log-likelihood over positive
        # (mask * log_prob).sum(1): log [sum(exp(zi*zp)) / sum(exp(zi*zl)) ]
        # mask.sum(1): |P(i)|
        # mean_log_prob_pos: L^sup_out
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss