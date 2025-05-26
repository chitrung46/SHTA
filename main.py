import os
import torch
import torch.nn as nn

from utils import parse_args, init_seed, Mkdir, print_model_parameters
from utils import scaler_mae_loss, scaler_rmse_loss
from dataloader import get_dataloader
from model import STHA
from trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args(device)
init_seed(args.seed, args.seed_mode)

#config log path
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'SAVE', args.dataset)
Mkdir(log_dir)
args.log_dir = log_dir

#load dataset
train_loader, val_loader, test_loader, scaler_data, scaler_day, scaler_week, scaler_holiday = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)
args.scaler_zeros = scaler_data.transform(0)
args.scaler_zeros_day = scaler_day.transform(0)
args.scaler_zeros_week = scaler_week.transform(0)

model = STHA(args, args.device, args.in_dim)
model = model.to(args.device)

if args.xavier:
    for p in model.parameters():
        if p.requires_grad==True:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

print_model_parameters(model, only_num=False)

if args.loss_func == 'mask_mae':
    loss = scaler_mae_loss(scaler_data, mask_value=args.mape_thresh)
    print('============================scaler_mae_loss')
elif args.loss_func == 'mask_rmse':
    loss = scaler_rmse_loss(scaler_data, mask_value=args.mape_thresh)
    print('============================scaler_rmse_loss')
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

loss_kl = nn.KLDivLoss(reduction='sum').to(args.device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)

trainer = Trainer(model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load(log_dir + '/best_model.pth'))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler_data, trainer.logger)
else:
    raise ValueError