import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

            
def create_effnetb2_model(num_classes:int, feature_extracting:bool):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # create weights object and transfrom from weights
    weights = EfficientNet_B2_Weights.DEFAULT
    transform = weights.transforms()
    
    # load effnetmodel with trained weights
    model = efficientnet_b2(weights=weights)
    
    # num features of the classifier
    in_features = list(model.classifier.children())[-1].in_features
    
    # freeze all layers
    #for param in model.parameters():
    #    param.requires_grad=False
    set_parameter_requires_grad(model, feature_extracting)
    
    # replace te old classifier head with a new one
    model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                 nn.Linear(in_features=in_features,out_features=num_classes))
    
    return model, transform
