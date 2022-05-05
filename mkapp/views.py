

import joblib
from torchvision import transforms
import torch.nn.functional as F
from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3 * 256 * 256, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view(-1, 3 * 256 * 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def input_form(request):

    # form の入力があった場合
    if request.method == 'POST' and request.FILES['pic']:

        image = request.FILES['pic']

        #loaded_model = torch.load('model/model.pt')

        image = Image.open(image)

        # 推論の実行
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # リサイズしてテンソルに変換
        image = transform(image)

        # ネットワークの準備
        net = Net().cpu().eval()

        # 重みの読み込み
        net.load_state_dict(torch.load('model/model.pt', map_location=torch.device('cpu')))

        y = net(image)
        y = torch.argmax(y)

        if y == 0:
            result = '佳奈'
        else:
            result = '茉奈'


        return render(request, 'result.html', {'y':result})

    # 通常のビュー
    else:
        #form = InputForm()

        return render(request, 'form.html')

