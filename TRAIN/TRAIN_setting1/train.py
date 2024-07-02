import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import os



curPath = os.path.abspath(os.path.dirname(__file__))


os.environ['CUDA_VISIBLE_DEVICES'] = '0'



from DATA.dataset import pre_dataset
from TRAIN.TRAIN_setting1.DA_MODEL import DANet
from UTILS import utils





if __name__ == '__main__':

    print('train da model')
    with open('train.yml', 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.FullLoader)

    patch_size = opt['patch_size']

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


    train_set = pre_dataset(img_dir=opt['dataset']['clean_dir'],patch_size=opt['patch_size'])
    train_loader = DataLoader(dataset=train_set, num_workers=opt['num_workers'], batch_size=opt['batch_size'], shuffle=True)

    DA_MODEL = DANet(opt)

    start_epoch = 1
    iteration = 1


    if opt['DANET_CHECKPOINT'] is not None:
        checkpoint = torch.load(opt['DANET_CHECKPOINT'])
        DA_MODEL.DP_MODEL.load_state_dict(checkpoint['model_DP_state_dict'])
        DA_MODEL.SRN_MODEL.load_state_dict(checkpoint['model_SRN_state_dict'])
        DA_MODEL.RB_MODEL.load_state_dict(checkpoint['model_RB_state_dict'])

        epoch = checkpoint['epoch']

        print('use SRN of epoch:', epoch)



    # prepare summary
    summary_path = ''
    if opt['saving']:
        save_path = os.path.join(curPath,opt['save_path'])
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path))
        dir_index = 0


        if opt['save_path'] is not None:
            summary_path = os.path.join(save_path, 'tb_logger')
        while os.path.isdir(os.path.join(summary_path, str(dir_index))):
            dir_index += 1
        summary_path = os.path.join(summary_path, str(dir_index))
        writer = SummaryWriter(summary_path)
        print('Saving summary into directory ' + summary_path + '/')

        ## save args
        args_path = os.path.join(summary_path, 'commandline_args.yml')

        with open(args_path, 'w',encoding='utf-8') as f:
            yaml.dump(data=opt, stream=f, allow_unicode=True)

    degrade = utils.SRMDPreprocessing(
        opt['scale'],
        kernel_size=opt['blur_kernel'],
        blur_type=opt['blur_type'],
        sig_min=opt['sig_min'],
        sig_max=opt['sig_max'],
        lambda_min=['lambda_min'],
        lambda_max=opt['lambda_max'],
        noise=opt['noise']
    )



    for epoch in range(start_epoch, opt['num_epochs'] + 1):
        train_bar = tqdm(train_loader, desc='[%d/%d]' % (epoch, opt['num_epochs']))

        DA_MODEL.Module_train()

        cnt = 0

        for hr in train_bar:

            iteration += 1
            cnt += 1

            hr = hr.cuda()  # b, n, c, h, w
            lr, b_kernels = degrade(hr)

            lr1 = lr[:, 0, ...]
            lr2 = lr[:, 1, ...]

            hr1 = hr[:, 0, ...]
            hr2 = hr[:, 1, ...]


            DA_MODEL.feed_data(lr1/255.0, hr1/255.0, lr2/255.0, hr2/255.0)


            if epoch <= opt['DA_TRAIN_epoch']:
                DA_MODEL.forward1()

                learning_rate_condition_branch = opt['DA_learning_rate'] * (
                            opt['gamma_DA'] ** (epoch // opt['DA_step_size']))

                for param_group in DA_MODEL.optimizer_DP.param_groups:
                    param_group['lr'] = learning_rate_condition_branch

                for param_group in DA_MODEL.optimizer_RB.param_groups:
                    param_group['lr'] = learning_rate_condition_branch

                DA_MODEL.update_Degradation_branch_alone()
            else:
                DA_MODEL.forward()

                if opt['pretrain']:
                    learning_rate_WHOLE = opt['learning_rate_SRN'] * (
                            opt['gamma_SRN'] ** ((epoch - opt['DA_TRAIN_epoch']) // opt['step_size']))
                else:
                    learning_rate_WHOLE = opt['learning_rate_SRN'] * (
                            opt['gamma_SRN'] ** (epoch // opt['step_size']))

                for param_group in DA_MODEL.optimizer_DP.param_groups:
                    param_group['lr'] = learning_rate_WHOLE

                for param_group in DA_MODEL.optimizer_RB.param_groups:
                    param_group['lr'] = learning_rate_WHOLE

                for param_group in DA_MODEL.optimizer_SRN.param_groups:
                    param_group['lr'] = learning_rate_WHOLE

                DA_MODEL.update_EG()


        print('lr:',DA_MODEL.optimizer_DP.param_groups[0]['lr'])
        print('LATENT index', DA_MODEL.latent.mean().item(), DA_MODEL.latent.std(unbiased=False).item())
        print('loss kld:', DA_MODEL.loss_kld.item())
        print('loss_c2_blur:', DA_MODEL.loss_c2_blur.item())
        print('loss_sr:', DA_MODEL.loss_sr.item())



        if opt['saving'] and epoch % opt['save_model_interval'] == 0 and epoch >= opt['DA_TRAIN_epoch']:
            path = os.path.join(save_path, 'checkpoints', 'epoch_{}.tar'.format(epoch))
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            DA_MODEL.model_save(path,epoch,save_path)








