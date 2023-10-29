import torch
from torch import nn
from sklearn.cluster import KMeans
import numpy as np


class EMA(nn.Module):
    def __init__(self, channels, num_classes, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups).cuda()
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0).cuda()
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1).cuda()
        self.num_classes = num_classes

    def ema_attention(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
    
    def cluster(self, bbox_pred):
        bbox_pred_xy_np = bbox_pred[:, :2].cpu().detach().numpy()
        kmeans = KMeans(n_clusters=self.num_classes)
        kmeans.fit(bbox_pred_xy_np)
        cluster_labels = kmeans.labels_
        each_num_class = [np.sum(cluster_labels<i) for i in range(self.num_classes)]
        index_num_class = [0 for i in range(self.num_classes)]
        bbox_pred_xy_np_cluster = [None for i in range(len(bbox_pred_xy_np))]
        mapping = []
        for i, (data_point, label) in enumerate(zip(bbox_pred, cluster_labels)):
            mapping.append(index_num_class[label]+each_num_class[label])
            bbox_pred_xy_np_cluster[index_num_class[label]+each_num_class[label]] = data_point
            index_num_class[label] += 1
        return mapping, torch.stack(bbox_pred_xy_np_cluster)

    def forward(self, bbox_feats, bbox_pred):
        mapping, bbox_pred = self.cluster(bbox_pred)
        bbox_feats = bbox_feats[mapping]
        bbox_feats = self.ema_attention(bbox_feats)
        return bbox_feats