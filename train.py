#先跑python crop_images.py  生成数据集.







import itertools
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.utils import save_image
import torchvision

from models import Generator, Discriminator

batch_size = 10
lambda_cycle = 1
lambda_identity = 2
lr = 0.0001
seed = 0

print_every = 200
n_epochs = 60
input_shape = (216, 176)
odir = 'ckpt'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

cudnn.benchmark = True

















# Init dataset
transformer = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(
            lambda img: img[:, 1:-1, 1:-1]),#裁剪掉最外边一圈像素
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

dataset = torchvision.datasets.ImageFolder('data/celeba/', transformer)
# dtatset.imgs 里面的数据情况是:(sample path, class_index)
labels_neg = [i for i, (_, l) in enumerate(dataset.imgs) if l == 0] #所以这个就是非笑容

labels_pos = [i for i, (_, l) in enumerate(dataset.imgs) if l == 1]#这个是笑容图片.

sampler_neg = torch.utils.data.sampler.SubsetRandomSampler(labels_neg)
sampler_pos = torch.utils.data.sampler.SubsetRandomSampler(labels_pos)

pos_loader = torch.utils.data.DataLoader(dataset,
                                         sampler=sampler_pos,
                                         batch_size=batch_size,
                                        )

neg_loader = torch.utils.data.DataLoader(dataset,
                                         sampler=sampler_neg,
                                         batch_size=batch_size,
                                        )

# Init models
netDP = Discriminator().cuda()
netDN = Discriminator().cuda()
netP2N = Generator().cuda()
netN2P = Generator().cuda()

criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss() #改成L2应该也差不多
criterion_gan = nn.MSELoss()

# Init tensors  #表示假的图片是从一个白板开始学起来的,最后生成假的图片.
real_pos = torch.zeros(batch_size, 3,
                       input_shape[0], input_shape[1]  #都是白板图片
                      ).cuda()
real_neg = torch.zeros(batch_size, 3,
                       input_shape[0], input_shape[1]
                      ).cuda()

real_lbl = torch.zeros(batch_size, 1).cuda()
real_lbl[:, 0] = 1  #表示是真图片
fake_lbl = torch.zeros(batch_size, 1).cuda()
fake_lbl[:, 0] = -1  #表示是假图片

opt_G = optim.Adam(list(netP2N.parameters())+list(netN2P.parameters()), lr=lr, betas = (0.5, 0.999))
opt_D = optim.Adam(list(netDN.parameters())+list(netDP.parameters()), lr=lr, betas = (0.5, 0.999))


netDN.train()
netDP.train()
netP2N.train()
netN2P.train()


scheduler_G = lr_scheduler.StepLR(opt_G, step_size=10, gamma=0.317)
scheduler_D = lr_scheduler.StepLR(opt_D, step_size=10, gamma=0.317)
print('Training...')  #下面的损失函数是gan算法的核心!!!!!!!!!!
for epoch in range(n_epochs):

    batch = 0

    for (pos, _), (neg, _) in zip(pos_loader, neg_loader):

        netDN.zero_grad()
        netDP.zero_grad()
        netP2N.zero_grad()
        netN2P.zero_grad()

        real_pos.copy_(pos)
        real_neg.copy_(neg)

        # Train P2N Generator
        real_pos_v = Variable(real_pos)
        fake_neg, mask_neg = netP2N(real_pos_v)    #p2n:真变假的网络, n2p是假变真的网络.
        rec_pos, _ = netN2P(fake_neg)  #刚生成的假图片,直接继续生成假的正例.
        fake_neg_lbl = netDN(fake_neg)  #DN 区分假图片的网络

        loss_P2N_cyc = criterion_cycle(rec_pos, real_pos_v)  #这个越小,越会造假
        loss_P2N_gan = criterion_gan(fake_neg_lbl, Variable(real_lbl))  #让生成器越来越能造假,让生成器越来越能干死识别器. 因为DN识别器给的结果,被gan用来越来越模拟原始的真实标签.
        loss_N2P_idnt = criterion_identity(fake_neg, real_pos_v)#保证生成的图片跟原始图片还是同一个人. 也就是让netP2N越来越不会改变主角身份信息.而只是笑容. 确实这个还是不好理解. 让神经网络自己去学, 我们定的目标是学习到保持身份信息,他学成啥样,自己慢慢迭代吧.






        # Train N2P Generator,跟上面类似
        real_neg_v = Variable(real_neg)
        fake_pos, mask_pos = netN2P(real_neg_v)
        rec_neg, _ = netP2N(fake_pos)
        fake_pos_lbl = netDP(fake_pos)  #区分真图片的网络

        loss_N2P_cyc = criterion_cycle(rec_neg, real_neg_v)
        loss_N2P_gan = criterion_gan(fake_pos_lbl, Variable(real_lbl))
        loss_P2N_idnt = criterion_identity(fake_pos, real_neg_v)




        #这个是生成器.假变真,真变假都让他越来越好,就是让区分器越来越识别不出来
        #下行只是加上权重而已,无脑过.

        loss_G = ((loss_P2N_gan + loss_N2P_gan)*0.5 +
                  (loss_P2N_cyc + loss_N2P_cyc)*lambda_cycle +
                  (loss_P2N_idnt + loss_N2P_idnt)*lambda_identity)

        loss_G.backward()
        opt_G.step()

        # Train Discriminators
        netDN.zero_grad()
        netDP.zero_grad()

        #D区分器    从真变到假的图片,让越来越能识别他是假图片
        '''
        detach
            官方文档中，对这个方法是这么介绍的。
            
            返回一个新的从当前图中分离的 Variable。
            返回的 Variable 永远不会需要梯度
        '''

        fake_neg_score = netDN(fake_neg.detach())#把fake_neg跟上面网络截断,用detach.不在影响上面网络的梯度.非常有用核心的语法!!!!!!!!!!!!!!!!
        loss_D = criterion_gan(fake_neg_score, Variable(fake_lbl))#识别器越来越会识别fake_neg图片,判定为假,也就是让生成的非笑容跟原始的非笑容找出区别.
        fake_pos_score = netDP(fake_pos.detach())
        loss_D += criterion_gan(fake_pos_score, Variable(fake_lbl))#让识别器越来越能
        # 找出生成的笑容和原始的笑容的区别

        real_neg_score = netDN.forward(real_neg_v)#同时保证对于真实的图片,处理之后还是真实的图片
        #因为你不能一顿学,连真人都识别错了,就太搞笑了.所以基本的图片分类器还是要学好.属于分类器的基本功.
        loss_D += criterion_gan(real_neg_score, Variable(real_lbl))
        real_pos_score = netDP.forward(real_pos_v)
        loss_D += criterion_gan(real_pos_score, Variable(real_lbl))
        #为了让导数小点,给G网络更长的学习时间.
        loss_D = loss_D*0.25

        loss_D.backward()
        opt_D.step()

        #
        scheduler_G.step()
        scheduler_D.step()
        #下面是保存数据用的.










        if batch % 10 == 0:

            print("跑了",batch)
        if batch % print_every == 0 and batch > 1:
            print('Epoch #%d' % (epoch+1))
            print('Batch #%d' % batch)

            print('Loss D: %0.3f' % loss_D.item()  + '\t' +
                  'Loss G: %0.3f' % loss_G.item() )
            print('Loss P2N G real: %0.3f' % loss_P2N_gan.item()  + '\t' +
                  'Loss N2P G fake: %0.3f' % loss_N2P_gan.item() )

            print('-'*50)
            sys.stdout.flush()

            save_image(torch.cat([
                real_neg.cpu()[0]*0.5+0.5,
                mask_pos.data.cpu()[0],
                fake_pos.data.cpu()[0]*0.5+0.5], 2),
                'progress_pos.png')
            save_image(torch.cat([
                real_pos.cpu()[0]*0.5+0.5,
                mask_neg.data.cpu()[0],
                fake_neg.data.cpu()[0]*0.5+0.5], 2)

                ,
                'progress_neg.png')

            torch.save(netN2P, odir+'/netN2P.pth')  #最后使用这个权重网络就会假变真
            torch.save(netP2N, odir+'/netP2N.pth')
            torch.save(netDN, odir+'/netDN.pth')
            torch.save(netDP, odir+'/netDP.pth')
        batch += 1
