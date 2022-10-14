CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method gp --gpu  0 --seed 77 &
pids[0]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method gp --gpu  1 --seed 78 &
pids[1]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method gp --gpu  0 --seed 79 &
pids[2]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method gp --gpu  4 --seed 77 &
pids[3]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method gp --gpu  3 --seed 78 &
pids[4]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method gp --gpu  3 --seed 79 &
pids[5]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method gp --gpu  1 --seed 77 &
pids[6]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method gp --gpu  4 --seed 78 &
pids[7]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method gp --gpu  0 --seed 79 &
pids[8]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method gp --gpu  1 --seed 77 &
pids[9]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method gp --gpu  5 --seed 78 &
pids[10]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method gp --gpu  5 --seed 79 &
pids[11]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method gp --gpu  5 --seed 77 &
pids[12]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method gp --gpu  3 --seed 78 &
pids[13]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method gp --gpu  5 --seed 79 &
pids[14]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method gp --gpu  4 --seed 77 &
pids[15]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method gp --gpu  4 --seed 78 &
pids[16]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method gp --gpu  3 --seed 79 &
pids[17]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.2 --method gp --gpu  4 --seed 77 &
pids[18]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.2 --method gp --gpu  5 --seed 78 &
pids[19]=$!
for pid in ${pids[*]}; 
do
	 wait $pid 
done
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.2 --method gp --gpu  1 --seed 79 &
pids[0]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.4 --method gp --gpu  4 --seed 77 &
pids[1]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.4 --method gp --gpu  0 --seed 78 &
pids[2]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.4 --method gp --gpu  4 --seed 79 &
pids[3]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.6 --method gp --gpu  4 --seed 77 &
pids[4]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.6 --method gp --gpu  4 --seed 78 &
pids[5]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type rcn --noise_strength 0.6 --method gp --gpu  5 --seed 79 &
pids[6]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.2 --method gp --gpu  4 --seed 77 &
pids[7]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.2 --method gp --gpu  4 --seed 78 &
pids[8]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.2 --method gp --gpu  4 --seed 79 &
pids[9]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.4 --method gp --gpu  1 --seed 77 &
pids[10]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.4 --method gp --gpu  4 --seed 78 &
pids[11]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.4 --method gp --gpu  0 --seed 79 &
pids[12]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.6 --method gp --gpu  1 --seed 77 &
pids[13]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.6 --method gp --gpu  3 --seed 78 &
pids[14]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset cifar10 --noise_type linear --noise_strength 0.6 --method gp --gpu  0 --seed 79 &
pids[15]=$!
