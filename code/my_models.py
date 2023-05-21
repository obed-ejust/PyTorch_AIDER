from torchvision import models
from torch import nn


def select_model(model_name, classes):
    model = None
    if model_name == "MobileNet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False

        # Modify the last layer
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, classes)

    elif model_name == "MobileNet_v3":
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False

        # Modify the last layer
        model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=classes, bias=True))

    elif model_name == "SqeezeNet1_0":
        model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights)
        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False
        # Modify the last layer
        model.classifier[1] = nn.Conv2d(512, classes, kernel_size=(1, 1), stride=(1, 1))

    elif model_name == "VGG16":
        model = models.vgg16(weights='IMAGENET1K_V1')
        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False
        # Modify the last layer
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=50, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=50, out_features=20, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=20, out_features=classes, bias=True)
        )
    elif model_name == "ShuffleNet_v2":
        model = models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(in_features=1024, out_features=classes, bias=True)

    elif model_name == "EfficientNet_B0":
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=classes, bias=True)

    elif model_name == "ResNet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(in_features=2048, out_features=classes, bias=True)

    else:
        print("Model not Defined")

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: " + str(total_params))
    print("Trainable_params: " + str(trainable_params))

    return model


