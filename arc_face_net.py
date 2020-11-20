from torch import nn
from torchvision import models
import cirtorch
import torch
from metric_learning import ArcMarginProduct


class ArcFaceNet(nn.Module):
    DIVIDABLE_BY = 32

    def __init__(self,
                 model_name='resnet50',
                 pooling=['GAP'],
                 args_pooling: dict = {},
                 fc_dim=512,
                 # dropout=0.0,
                 pretrained=False,
                 class_num=None):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ArcFaceNet, self).__init__()

        if model_name.startswith("efficientnet"):
            from efficientnet_pytorch import EfficientNet
            backbone = EfficientNet.from_pretrained(model_name, num_classes=1000)
            last_linear_idx = -2
            pool_idx = -4
            backbone_layers = list(backbone.children())
            final_in_features = backbone_layers[last_linear_idx].in_features
            self.backbone = backbone
        elif model_name.startswith('bagnet'):
            import bagnets.pytorchnet
            backbone = getattr(bagnets.pytorchnet, model_name)(pretrained=pretrained)
            last_linear_idx = -1
            pool_idx = -2

            backbone_layers = list(backbone.children())
            final_in_features = backbone_layers[last_linear_idx].in_features
            self.backbone = nn.Sequential(*backbone_layers[:pool_idx])
        else:
            backbone = getattr(models, model_name)(num_classes=1000)
            last_linear_idx = -1
            pool_idx = -2

            backbone_layers = list(backbone.children())
            final_in_features = backbone_layers[last_linear_idx].in_features
            self.backbone = nn.Sequential(*backbone_layers[:pool_idx])

        self.pooling_param = pooling
        if len(pooling) == 1:
            if pooling[0] == 'GAP':
                self.pooling = nn.AdaptiveAvgPool2d(1)
            else:
                self.pooling = getattr(cirtorch.pooling, pooling[0])(**args_pooling)
        else:
            pooling_list = []
            for p in pooling:
                if p == 'GAP':
                    pooling_list.append(nn.AdaptiveAvgPool2d(1))
                else:
                    pooling_list.append(getattr(cirtorch.pooling, p)(**args_pooling))
            final_in_features *= len(pooling)
            self.pooling = nn.ModuleList(pooling_list)

        # self.pooling = getattr(cirtorch.pooling, pooling)(**args_pooling)

        # self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(final_in_features)
        self.fc1 = nn.Linear(final_in_features, fc_dim)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.bn3 = nn.BatchNorm1d(fc_dim)
        self.last_fc = nn.Linear(fc_dim, class_num)

        self.arc = ArcMarginProduct(fc_dim, class_num,
                                    s=30.0, m=0.3, easy_margin=False, ls_eps=0.0)

        self.model_name = model_name

    def forward(self, x, label=None):
        return self.extract_feat(x, label)

    def extract_feat(self, x, label=None):
        batch_size = x.shape[0]
        if self.model_name.startswith("efficientnet"):
            x = self.backbone.extract_features(x)
        else:
            x = self.backbone(x)

        if len(self.pooling_param) == 1:
            x = self.pooling(x).view(batch_size, -1)
        else:
            pool_out_list = []
            for p in self.pooling:
                pool_out_list.append(p(x).view(batch_size, -1))
            x = torch.cat(pool_out_list, dim=1)

        # x = self.dropout(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.bn3(x)

        if label is None:
            return self.last_fc(x)

        arc_output = self.arc(x, label)
        logits = self.last_fc(x)

        return arc_output, logits
