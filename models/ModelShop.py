import torch.nn as nn
from .MLP        import MLP
from .ResNet     import ResNet18BN, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .ResNet_AP  import ResNet18_AP, ResNet18BN_AP
from .VGG        import VGG11BN, VGG11, VGG13, VGG16, VGG19

def ModelShop(
    M_model_name:   str = "AlexNet",
    I_image_channel:int = 3,
    I_image_size:   list = [32,32],
    O_num_classes:  int = 10,
    N_norm:         str = "",
    N_act:          str = "",
    N_pool:         str = "",
    N_width:        int = 0,
    N_depth:        int = 0,
    A_param1:       str = "",
    A_param2:       str = "",
    A_param3:       str = ""
):
    Model: nn.Module

    if M_model_name == "MLP":
        Model = MLP(I_image_channel*I_image_size[0]*I_image_size[1], O_num_classes)
    elif M_model_name == "ResNet18BN": Model = ResNet18BN(I_image_channel, O_num_classes)
    elif M_model_name == "ResNet18": Model = ResNet18(I_image_channel, O_num_classes)
    elif M_model_name == "ResNet34": Model = ResNet34(I_image_channel, O_num_classes)
    elif M_model_name == "ResNet50": Model = ResNet50(I_image_channel, O_num_classes)
    elif M_model_name == "ResNet101": Model = ResNet101(I_image_channel, O_num_classes)
    elif M_model_name == "ResNet152": Model = ResNet152(I_image_channel, O_num_classes)
    elif M_model_name == "ResNet18_AP": Model = ResNet18_AP(I_image_channel, O_num_classes)
    elif M_model_name == "ResNet18BN_AP": Model = ResNet18BN_AP(I_image_channel, O_num_classes)
    elif M_model_name == "VGG11BN": Model = VGG11BN(I_image_channel, O_num_classes)
    elif M_model_name == "VGG11": Model = VGG11(I_image_channel, O_num_classes)
    elif M_model_name == "VGG13": Model = VGG13(I_image_channel, O_num_classes)
    elif M_model_name == "VGG16": Model = VGG16(I_image_channel, O_num_classes)
    elif M_model_name == "VGG19": Model = VGG19(I_image_channel, O_num_classes)
    
    return Model