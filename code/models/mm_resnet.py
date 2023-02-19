"""
Script containing the multi-branch resnet class.
"""

import torch
from torch.nn import Module, Sequential
from models.my_resnet import resnet18, resnet50, BasicBlock, Bottleneck

class ResNetFuse(Module):
    """
    Multimodal ResNet with source-wise input branches and concatenation fusion.

    res_type (str)          :   ResNet Type (resnet18 or resnet50)
    fusoin_stage (int)      :   Stage of fusion (0,1,2,3,4,5)
    branch_split (list)     :   How to split input channels into branches
    num_classes (int)       :   Number of classes to classify
    device (torch.device)   :   Computation device ("cpu" or "gpu")
    """
    def __init__(self,
                 res_type : str,
                 fusion_stage : int,
                 branch_split : list[int],
                 num_classes : int,
                 device : torch.device):
        super().__init__()

        assert res_type in ["resnet18", "resnet50"]
        assert len(branch_split) == 2, "only two modality case is considered"
        if res_type == "resnet18":
            model_list = [resnet18(), resnet18()]
            fused_in_size = [64, 256, 512, 1024, int(32768/4)][fusion_stage-1]
            self.block = BasicBlock

        elif res_type == "resnet50":
            model_list = [resnet50(), resnet50()]
            fused_in_size = [64, 256, 512, 1024, 32768][fusion_stage-1]
            self.block = Bottleneck

        self.num_branches = len(branch_split)
        self.num_classes = num_classes
#
        self.branch_split = branch_split

        self.branch_list = []

        fused_in_size = fused_in_size * 2

        # Build source-wise feature extractor branches (0 <= fusion_stage <= 5)
        if fusion_stage == 0:
            self.branch_list.append(Sequential(torch.nn.Identity()))
            self.branch_list.append(Sequential(torch.nn.Identity()))
            self.fused_part = Sequential(
                                        torch.nn.Conv2d(
                                            in_channels=self.branch_split[0]+self.branch_split[1],
                                            out_channels=64,
                                            kernel_size=(7, 7),
                                            stride=(2, 2),
                                            padding=(3, 3),
                                            device=device,
                                            bias=False),
                                        model_list[0].bn1,
                                        model_list[0].maxpool,
                                        model_list[0].layer1,
                                        model_list[0].layer2,
                                        model_list[0].layer3,
                                        model_list[0].layer4,
                                        model_list[0].avgpool,
                                        torch.nn.Flatten(start_dim=1, end_dim=-1),
                                        model_list[0].fc,
                                        torch.nn.Linear(in_features=model_list[0].fc.out_features,
                                                        out_features=self.num_classes,
                                                        bias=True, device=device))
        else:
            for i, in_channels in enumerate(branch_split):
                if fusion_stage == 1:
                    self.branch_list.append(Sequential(
                                                torch.nn.Conv2d(
                                                    in_channels=in_channels,
                                                    out_channels=64,
                                                    kernel_size=(7, 7),
                                                    stride=(2, 2),
                                                    padding=(3, 3),
                                                    device=device,
                                                    bias=False),
                                            model_list[i].bn1,
                                            model_list[i].maxpool
                                            )
                                            )

                elif fusion_stage == 2:
                    self.branch_list.append(Sequential(
                                                torch.nn.Conv2d(
                                                    in_channels=in_channels,
                                                    out_channels=64,
                                                    kernel_size=(7, 7),
                                                    stride=(2, 2),
                                                    padding=(3, 3),
                                                    device=device,
                                                    bias=False),
                                            model_list[i].bn1,
                                            model_list[i].maxpool,
                                            model_list[i].layer1
                                            )
                                            )

                elif fusion_stage == 3:
                    self.branch_list.append(Sequential(
                                                torch.nn.Conv2d(
                                                    in_channels=in_channels,
                                                    out_channels=64,
                                                    kernel_size=(7, 7),
                                                    stride=(2, 2),
                                                    padding=(3, 3),
                                                    device=device,
                                                    bias=False),
                                            model_list[i].bn1,
                                            model_list[i].maxpool,
                                            model_list[i].layer1,
                                            model_list[i].layer2
                                            )
                                            )

                elif fusion_stage == 4:
                    self.branch_list.append(Sequential(
                                                torch.nn.Conv2d(
                                                    in_channels=in_channels,
                                                    out_channels=64,
                                                    kernel_size=(7, 7),
                                                    stride=(2, 2),
                                                    padding=(3, 3),
                                                    device=device,
                                                    bias=False),
                                            model_list[i].bn1,
                                            model_list[i].maxpool,
                                            model_list[i].layer1,
                                            model_list[i].layer2,
                                            model_list[i].layer3
                                            )
                                            )

                elif fusion_stage == 5:
                    self.branch_list.append(Sequential(
                                                torch.nn.Conv2d(
                                                    in_channels=in_channels,
                                                    out_channels=64,
                                                    kernel_size=(7, 7),
                                                    stride=(2, 2),
                                                    padding=(3, 3),
                                                    device=device,
                                                    bias=False),
                                            model_list[i].bn1,
                                            model_list[i].maxpool,
                                            model_list[i].layer1,
                                            model_list[i].layer2,
                                            model_list[i].layer3,
                                            model_list[i].layer4
                                            )
                                            )


            # Build combined part where concatenated sources are processed.
            if fusion_stage == 1:
                para = list(model_list[0].layer1.children())[0]
                self.fused_part = Sequential(
                                    self.block(
                                        inplanes=para.inplanes*2,
                                        planes = para.planes,
                                        stride = para.stride,
                                        downsample= torch.nn.Sequential(
                                                        torch.nn.Conv2d(
                                                            in_channels=2 * para.inplanes,
                                                            out_channels=para.planes*\
                                                                         para.expansion,
                                                            stride=para.stride,
                                                            kernel_size=1,
                                                            bias=False),
                                                            para.norm_layer(para.planes*
                                                                            para.expansion)),
                                            groups=para.groups,
                                            base_width=para.base_width,
                                            dilation=para.dilation,
                                            norm_layer = para.norm_layer
                                            ),
                                    *list(model_list[0].layer1.children())[1:],
                                    model_list[0].layer2,
                                    model_list[0].layer3,
                                    model_list[0].layer4,
                                    model_list[0].avgpool,
                                    torch.nn.Flatten(start_dim=1, end_dim=-1),
                                    model_list[0].fc,
                                    torch.nn.Linear(in_features=model_list[0].fc.out_features,
                                                    out_features=self.num_classes,
                                                    bias=True, device=device))
            elif fusion_stage == 2:
                para = list(model_list[0].layer2.children())[0]
                self.fused_part = Sequential(
                                    self.block(
                                        inplanes=2 * para.inplanes,
                                        planes = para.planes,
                                        stride = para.stride,
                                        downsample= torch.nn.Sequential(
                                                        torch.nn.Conv2d(
                                                            in_channels=2*para.inplanes,
                                                            out_channels=para.planes * para.expansion,
                                                            stride=para.stride,
                                                            kernel_size=1,
                                                            bias=False),
                                        para.norm_layer(para.planes * para.expansion)),
                                        groups=para.groups,
                                        base_width=para.base_width,
                                        dilation=para.dilation,
                                        norm_layer = para.norm_layer
                                        ),
                                    *list(model_list[0].layer2.children())[1:],
                                    model_list[0].layer3,
                                    model_list[0].layer4,
                                    model_list[0].avgpool,
                                    torch.nn.Flatten(start_dim=1, end_dim=-1),
                                    model_list[0].fc,
                                    torch.nn.Linear(in_features=model_list[0].fc.out_features,
                                                    out_features=self.num_classes,
                                                    bias=True, device=device)
                                    )
            elif fusion_stage == 3:
                para = list(model_list[0].layer3.children())[0]
                self.fused_part = Sequential(
                                    self.block(
                                        inplanes=2 * para.inplanes,
                                        planes = para.planes,
                                        stride = para.stride,
                                        downsample= torch.nn.Sequential(
                                                        torch.nn.Conv2d(
                                                            in_channels=2*para.inplanes,
                                                            out_channels=para.planes*para.expansion,
                                                            stride=para.stride,
                                                            kernel_size=1,
                                                            bias=False),
                                        para.norm_layer(para.planes * para.expansion)),
                                        groups=para.groups,
                                        base_width=para.base_width,
                                        dilation=para.dilation,
                                        norm_layer = para.norm_layer
                                        ),
                                    *list(model_list[0].layer3.children())[1:],
                                    model_list[0].layer4,
                                    model_list[0].avgpool,
                                    torch.nn.Flatten(start_dim=1, end_dim=-1),
                                    model_list[0].fc,
                                    torch.nn.Linear(in_features=model_list[0].fc.out_features,
                                                    out_features=self.num_classes,
                                                    bias=True, device=device)
                                    )
            elif fusion_stage == 4:
                para = list(model_list[0].layer4.children())[0]
                self.fused_part = Sequential(
                                    self.block(
                                        inplanes=2 * para.inplanes,
                                        planes = para.planes,
                                        stride = para.stride,
                                        downsample= torch.nn.Sequential(
                                                        torch.nn.Conv2d(
                                                            in_channels=2*para.inplanes,
                                                            out_channels=para.planes*para.expansion,
                                                            stride=para.stride,
                                                            kernel_size=1,
                                                            bias=False),
                                                        para.norm_layer(para.planes*para.expansion)
                                                        ),
                                        groups=para.groups,
                                        base_width=para.base_width,
                                        dilation=para.dilation,
                                        norm_layer = para.norm_layer
                                        ),
                                    *list(model_list[0].layer4.children())[1:],
                                    model_list[0].avgpool,
                                    torch.nn.Flatten(start_dim=1, end_dim=-1),
                                    model_list[0].fc,
                                    torch.nn.Linear(in_features=model_list[0].fc.out_features,
                                                    out_features=self.num_classes,
                                                    bias=True, device=device)
                                    )
            elif fusion_stage == 5:
                self.fused_part = Sequential(
                                    model_list[0].avgpool,
                                    torch.nn.Flatten(start_dim=1, end_dim=-1),
                                    torch.nn.Linear(in_features=model_list[0].fc.in_features * 2,
                                                    out_features=1000,
                                                    bias=True, device=device),
                                    torch.nn.Linear(in_features=1000,
                                                    out_features=self.num_classes,
                                                    bias=True, device=device)
                                    )

        self.branch_list=torch.nn.ModuleList(self.branch_list)


    def forward(self, x, give_branch_out=False, multishot=True) -> torch.Tensor:
        """
        Forward propagation of input x.

        x (torch.tensor)        :   Tensor containing input images
        give_branch_out (bool)  :   False: return only prediction
                                    True: return prediction and source features
        """
        out_list = []

        if len(self.branch_split) > 1:
            in_list = torch.split(x, self.branch_split, dim=-3)

            for i, x_branch in enumerate(in_list):
                out_list.append(self.branch_list[i](x_branch))
        else:
            out_list.append(self.branch_list[0](x))

        if multishot:
            x_cat = torch.cat(
                [2*(torch.sigmoid(
                        torch.cat([out_list[0],out_list[1]], dim=1))-0.5),         # both modalities
                        torch.cat([2*(torch.sigmoid(out_list[0])-0.5),
                                    torch.zeros_like(out_list[1])], dim=1),        # SAR only
                        torch.cat([torch.zeros_like(out_list[0]),
                                    2*(torch.sigmoid(out_list[1])-0.5)], dim=1)],  # Optical only
                dim=0)


        else:
            x_cat = 2*(torch.sigmoid(torch.cat([out_list[0],out_list[1]], dim=1))-0.5)

        pred = self.fused_part(x_cat)

        if give_branch_out:
            return pred, out_list[0], out_list[1]

        return pred
