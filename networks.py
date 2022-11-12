import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# refer: https://github.com/liuyugeng/ML-Doctor
class ShadowAttackModel(nn.Module):
	# input: output + prediction double-way
	# output: 2
	def __init__(self, class_num):
		super(ShadowAttackModel, self).__init__()
		self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(128, 256),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, prediction):
		Output_Component_result = self.Output_Component(output)
		Prediction_Component_result = self.Prediction_Component(prediction)
		
		final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), 1)
		final_result = self.Encoder_Component(final_inputs)

		return final_result

class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10, dropout=0.0):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
			nn.Dropout(p=dropout), # often used before FC-layer
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
			nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
			# nn.Softmax(dim=1), # 0 column  1 row
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class DiscreteClassifier(nn.Module):
    def __init__(self,num_feature, num_cls, dropout=0.0):
        super(DiscreteClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(num_feature,1024),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128, num_cls)
        
    def forward(self,x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)

# refer: https://pytorch.org/vision/main/_modules/torchvision/models/alexnet.html
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, dim_in: int = 3) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(dim_in, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class other_models(nn.Module):
  def __init__(self, model_name, in_channel=3, cls_num=10, dropout=0.0):
    super().__init__()
    # define model and loss
    _models = {
        'vgg16': models.vgg16(num_classes=cls_num),
        'alex':  AlexNet(num_classes=cls_num, dropout=dropout, dim_in=in_channel),
        'densenet': models.densenet121(num_classes=cls_num, drop_rate=dropout),
        'resnet18': models.resnet18(num_classes=cls_num),
        'resnet152': models.resnet152(num_classes=cls_num)
    }
    self.model = _models[model_name]
    if model_name.find('resnet') != -1 and in_channel == 1:
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

  def forward(self, x):
    return F.softmax(self.model(x), dim=1)

def model_type(model_name='cnn', in_channel=3, cls_num=10, dropout=0.0):
    _models = ['vgg16', 'alex', 'densenet', 'resnet18', 'resnet152']
    if model_name == 'cnn':
        _model = CNN(in_channel, cls_num, dropout)
    elif model_name == 'mlp':
        _model = DiscreteClassifier(in_channel, cls_num, dropout) # in_channel: input_size
    elif model_name in _models:
        _model = other_models(model_name, in_channel, cls_num, dropout)
    return _model