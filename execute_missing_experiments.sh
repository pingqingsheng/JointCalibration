CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method gp --gpu  0 --seed 77 &
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method gp --gpu  1 --seed 78 &
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method gp --gpu  4 --seed 79 &
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method gp --gpu  5 --seed 77 &
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method gp --gpu  0 --seed 78 &
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method gp --gpu  1 --seed 79 &

CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.2 --method gp --gpu  4 --seed 77 &
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.4 --method gp --gpu  5 --seed 78 &
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.6 --method gp --gpu  0 --seed 79 &
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.2 --method gp --gpu  1 --seed 77 &
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.4 --method gp --gpu  4 --seed 78 &
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.6 --method gp --gpu  5 --seed 79 &
