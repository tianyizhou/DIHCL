# DIHCL-Exp, dLoss
CUDA_VISIBLE_DEVICES=1,0 python dihcl.py -d cifar10 -a wrn --epochs 300 --wd 1e-4 --k 1.0 --dk 0.1 --mk 0.3 --use_curriculum --bandits_alg 'EXP3' --use_mean_teacher --gpu-id '1,0' --data_path ../data --save_path checkpoints/cifar10/AdaAug1 --trialID 'dihcl_exp_dloss_wrn2810/'
CUDA_VISIBLE_DEVICES=1,0 python dihcl.py -d cifar100 -a wrn --epochs 300 --wd 2e-4 --k 1.0 --dk 0.1 --mk 0.4 --use_curriculum --bandits_alg 'EXP3' --use_mean_teacher --gpu-id '1,0' --data_path ../data --save_path checkpoints/cifar100/AdaAug1 --trialID 'dihcl_exp_dloss_wrn2810/'

# DIHCL-Exp, Loss
CUDA_VISIBLE_DEVICES=1,0 python dihcl.py -d cifar10 -a wrn --epochs 300 --wd 1e-4 --k 1.0 --dk 0.1 --mk 0.3 --use_curriculum --bandits_alg 'EXP3' --use_mean_teacher --use_loss_as_feedback --gpu-id '1,0' --data_path ../data --save_path checkpoints/cifar10 --trialID 'dihcl_exp_loss_wrn2810/'
CUDA_VISIBLE_DEVICES=1,0 python dihcl.py -d cifar100 -a wrn --epochs 300 --wd 2e-4 --k 1.0 --dk 0.1 --mk 0.4 --use_curriculum --bandits_alg 'EXP3' --use_mean_teacher --use_loss_as_feedback --gpu-id '1,0' --data_path ../data --save_path checkpoints/cifar100 --trialID 'dihcl_exp_loss_wrn2810/'

# DIHCL-UCB, dLoss
CUDA_VISIBLE_DEVICES=1,0 python dihcl.py -d cifar10 -a wrn --epochs 300 --wd 1e-4 --k 1.0 --dk 0.1 --mk 0.3 --use_curriculum --bandits_alg 'UCB' --use_mean_teacher --gpu-id '1,0' --data_path ../data --save_path checkpoints/cifar10/AdaAug1 --trialID 'dihcl_ucb_dloss_wrn2810/'
CUDA_VISIBLE_DEVICES=1,0 python dihcl.py -d cifar100 -a wrn --epochs 300 --wd 2e-4 --k 1.0 --dk 0.1 --mk 0.4 --use_curriculum --bandits_alg 'UCB' --use_mean_teacher --gpu-id '1,0' --data_path ../data --save_path checkpoints/cifar100/AdaAug1 --trialID 'dihcl_ucb_dloss_wrn2810/'

# DIHCL-UCB, loss
CUDA_VISIBLE_DEVICES=1,0 python dihcl.py -d cifar10 -a wrn --epochs 300 --wd 1e-4 --k 1.0 --dk 0.1 --mk 0.3 --use_curriculum --bandits_alg 'UCB' --use_mean_teacher --use_loss_as_feedback --gpu-id '1,0' --data_path ../data --save_path checkpoints/cifar10 --trialID 'dihcl_ucb_loss_wrn2810/'
CUDA_VISIBLE_DEVICES=1,0 python dihcl.py -d cifar100 -a wrn --epochs 300 --wd 2e-4 --k 1.0 --dk 0.1 --mk 0.4 --use_curriculum --bandits_alg 'UCB' --use_mean_teacher --use_loss_as_feedback --gpu-id '1,0' --data_path ../data --save_path checkpoints/cifar100 --trialID 'dihcl_ucb_loss_wrn2810/'

# DIHCL-Beta, Flip
CUDA_VISIBLE_DEVICES=1,0 python dihcl.py -d cifar10 -a wrn --epochs 300 --wd 1e-4 --k 1.0 --dk 0.1 --mk 0.3 --use_curriculum --bandits_alg 'TS' --use_mean_teacher --gpu-id '1,0' --data_path ../data --save_path checkpoints/cifar10/AdaAug1 --trialID 'dihcl_ts_flip_wrn2810/'
CUDA_VISIBLE_DEVICES=1,0 python dihcl.py -d cifar100 -a wrn --epochs 300 --wd 2e-4 --k 1.0 --dk 0.1 --mk 0.4 --use_curriculum --bandits_alg 'TS' --use_mean_teacher --gpu-id '1,0' --data_path ../data --save_path checkpoints/cifar100/AdaAug1 --trialID 'dihcl_ts_flip_wrn2810/'