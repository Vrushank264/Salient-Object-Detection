import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from U2Net import U2Net


def show(model_path, img_path, attn_map = True):
    
    model = U2Net().cuda()
    model.load_state_dict(torch.load(model_path))
    
    img = T.ToTensor()(Image.open(img_path).convert('RGB'))
    img = img.unsqueeze(0).cuda()
    mask, _, _, _, _, _, _ = model(img)
    
    img = img.squeeze(0)
    img = img.permute(1,2,0)
    img = img.detach().cpu().numpy()
    mask = mask.squeeze(0).squeeze(0)
    
    if attn_map is False:
        
        mask = (mask > 0.5).float()
    
    mask = mask.detach().cpu().numpy()
    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(img)
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(mask, cmap = 'gray')
    plt.show()
    
    
if __name__ == '__main__':
    
    
    model_path = '/content/drive/MyDrive/U2Net/U2net_model.pth'
    img_path = '/content/Virat-Kohli.jpg'
    show(model_path, img_path)