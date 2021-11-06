import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from U2Net import U2Net
from dataset import Data


def compute_loss(bce_loss, s0, s1, s2, s3, s4, s5, s6, labels, w = 1.0):
    
    l0 = bce_loss(s0, labels)
    l1 = bce_loss(s1, labels)
    l2 = bce_loss(s2, labels)
    l3 = bce_loss(s3, labels)
    l4 = bce_loss(s4, labels)
    l5 = bce_loss(s5, labels)
    l6 = bce_loss(s6, labels)
    
    loss = w * (l0 + l1 + l2 + l3 + l4 + l5 + l6)
    
    return l0, loss


def train(loader, model, opt, bce_loss, writer, epoch, step, device = torch.device('cuda')):
    
    model.train()
    epoch_loss, epoch_loss0 = 0.0, 0.0
    num_correct, num_pixels = 0, 0
    train_loss, train_loss0 = [], []
    loop = tqdm(loader, leave = True, position = 0)
    
    for i, (img, mask) in enumerate(loop):
        
        img = img.type(torch.FloatTensor)
        mask = mask.type(torch.FloatTensor)
        
        inputs, labels = Variable(img.to(device), requires_grad = False), Variable(mask.to(device), requires_grad = False)
        
        opt.zero_grad()
        s0, s1, s2, s3, s4, s5, s6 = model(inputs)
        loss0, loss = compute_loss(bce_loss, s0, s1, s2, s3, s4, s5, s6, labels)
        pred = (s0 > 0.5).float()
        num_correct += (pred == labels).sum()
        num_pixels += torch.numel(pred) 
        train_loss.append(loss.item())
        train_loss0.append(loss0.item())
        loss.backward()
        opt.step()
        
        epoch_loss = sum(train_loss) / len(train_loss)
        epoch_loss0 = sum(train_loss0) / len(train_loss0)
            
        del s0, s1, s2, s3, s4, s5, s6, loss0, loss
        
        writer.add_scalar("Training/Loss", epoch_loss, global_step = step)
        writer.add_scalar("Training/Loss0", epoch_loss0, global_step = step)
        
        if i % 200 == 0:
        
            model.eval()
            with torch.no_grad():
                gen_label, _, _, _, _, _, _ = model(fixed_ip)
                gen_label_float = (gen_label > 0.5).float()

                writer.add_image("Observation/gen_mask", gen_label.squeeze(0), global_step = step)
                writer.add_image("Observation/gen_mask_0.5", gen_label_float.squeeze(0), global_step = step)
                
        torch.cuda.empty_cache()
        step += 1
        
    print(f'\nGot {num_correct}/{num_pixels} with acc {(num_correct/num_pixels) * 100:.3f}.')
    print(f"\n Train Loss: {epoch_loss}, loss0: {epoch_loss0}")

    return step

    
def test(loader, model, bce_loss, writer, epoch, test_step, device = torch.device('cuda')):
    
    test_l, test_l0 = 0.0, 0.0
    num_correct, num_pixels = 0, 0
    test_loss, test_loss0 = [], []
    loop = tqdm(loader, leave = True, position = 0)
    model.eval()
    
    with torch.no_grad():
        
        for i, (img, mask) in enumerate(loop):
            
            img = img.type(torch.FloatTensor)
            mask = mask.type(torch.FloatTensor)
        
            inputs, labels = Variable(img.to(device), requires_grad = False), Variable(mask.to(device), requires_grad = False)
            s0, s1, s2, s3, s4, s5, s6 = model(inputs)
            pred = (s0 > 0.5).float()
            num_correct += (pred == labels).sum()
            num_pixels += torch.numel(pred)
            loss0, loss = compute_loss(bce_loss, s0, s1, s2, s3, s4, s5, s6, labels)
            test_loss.append(loss.item())
            test_loss0.append(loss0.item())
            
            test_step += 1
        
            
        test_l = sum(test_loss) / len(test_loss)
        test_l0 = sum(test_loss0) / len(test_loss0)
        
        writer.add_scalar("Test/Loss", test_l, global_step = test_step)
        writer.add_scalar("Test/Loss0", test_l0, global_step = test_step)
        
        gen_mask, _, _, _, _, _, _ = model(fixed_ip_test)
        gen_mask_float = (gen_mask > 0.5).float()
        writer.add_image("Test Observation/Mask", gen_mask_float.squeeze(0), global_step = test_step)
        writer.add_image("Test Observation/Mask1", gen_mask.squeeze(0), global_step = test_step)
        vutils.save_image(gen_mask, f'/content/drive/MyDrive/U2Net/images/test/img{epoch}_{i}.png', normalize = True)

        print(f'Got {num_correct}/{num_pixels} with acc {(num_correct/num_pixels) * 100:.3f}.')
        print(f"\nTest Loss: {test_l}")
        return test_step
    
    
def main():
    
    TRAIN_IMG_DIR = '/content/DUTS-TR/DUTS-TR-Image'
    TRAIN_MASK_DIR = '/content/DUTS-TR/DUTS-TR-Mask'
    TEST_IMG_DIR = '/content/DUTS-TE/DUTS-TE-Image'
    TEST_MASK_DIR = '/content/DUTS-TE/DUTS-TE-Mask'
    LOGS_DIR = '/content/drive/MyDrive/U2Net/logs'
    MODEL_SAVE_DIR = '/content/drive/MyDrive/U2Net'
    device = torch.device('cuda')

    model = U2Net().to(device)
    #model.load_state_dict(torch.load('/content/drive/MyDrive/U2Net/U2net_model.pth'))
    writer = SummaryWriter(LOGS_DIR)
    step, test_step = 0, 0
    
    train_data = Data(img_dir = TRAIN_IMG_DIR, label_dir = TRAIN_MASK_DIR)
    test_data = Data(img_dir = TEST_IMG_DIR, label_dir = TEST_MASK_DIR, mode = 'test')
    train_loader = DataLoader(train_data, batch_size = 12, shuffle = True, num_workers = 2)
    test_loader = DataLoader(test_data, batch_size = 16, shuffle = False)
    
    bce_loss = nn.BCELoss(size_average = True)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3, betas = (0.9,0.999), eps = 1e-8, weight_decay = 0)
    
    global fixed_ip
    fixed_ip = T.ToTensor()(Image.open('/content/DUTS-TR/DUTS-TR-Image/ILSVRC2012_test_00001213.jpg').convert('RGB')).unsqueeze(0)
    fixed_ip = fixed_ip.to(device)
    
    global fixed_ip_test
    fixed_ip_test = T.ToTensor()(Image.open('/content/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00002362.jpg').convert('RGB')).unsqueeze(0)
    fixed_ip_test = fixed_ip_test.to(device)
    
    target = T.ToTensor()(Image.open('/content/DUTS-TR/DUTS-TR-Mask/ILSVRC2012_test_00001213.png').convert('L'))
    target = target.to(device)
    target_test = T.ToTensor()(Image.open('/content/DUTS-TE/DUTS-TE-Mask/ILSVRC2012_test_00002362.png').convert('L'))
    target_test = target_test.to(device)
    
    writer.add_image("Observation/input", fixed_ip.squeeze(0), global_step = 0)
    writer.add_image("Observation/target", target, global_step = 0)
    writer.add_image("Test Observation/input", fixed_ip_test.squeeze(0), global_step = 0)
    writer.add_image("Test Observation/target", target_test, global_step = 0)
    
    for epoch in range(1, 25):
        
        print("#-------------------------------------------------------------#")
        print("Epoch: ", epoch)
        print("#-------------------------------------------------------------#")
        step = train(train_loader, model, opt, bce_loss, writer, epoch, step)
        torch.save(model.state_dict(), open(MODEL_SAVE_DIR + '/U2net_model.pth', 'wb'))
        
        if epoch % 5 == 0:
            
            test_step = test(test_loader, model, bce_loss, writer, epoch, test_step)
    
    
if __name__ == '__main__':

    main()    