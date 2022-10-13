CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ts+oursv1 --gpu  0 --seed 77 &
pids[0]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ts+oursv1 --gpu  2 --seed 78 &
pids[1]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ts+oursv1 --gpu  5 --seed 79 &
pids[2]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ts+oursv1 --gpu  4 --seed 77 &
pids[3]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ts+oursv1 --gpu  2 --seed 78 &
pids[4]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ts+oursv1 --gpu  0 --seed 79 &
pids[5]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ts+oursv1 --gpu  4 --seed 77 &
pids[6]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ts+oursv1 --gpu  2 --seed 78 &
pids[7]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ts+oursv1 --gpu  1 --seed 79 &
pids[8]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ts+oursv1 --gpu  1 --seed 77 &
pids[9]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ts+oursv1 --gpu  3 --seed 78 &
pids[10]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ts+oursv1 --gpu  0 --seed 79 &
pids[11]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ts+oursv1 --gpu  3 --seed 77 &
pids[12]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ts+oursv1 --gpu  2 --seed 78 &
pids[13]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ts+oursv1 --gpu  0 --seed 79 &
pids[14]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ts+oursv1 --gpu  5 --seed 77 &
pids[15]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ts+oursv1 --gpu  0 --seed 78 &
pids[16]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ts+oursv1 --gpu  3 --seed 79 &
pids[17]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+mcdrop+oursv1 --gpu  5 --seed 77 &
pids[18]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+mcdrop+oursv1 --gpu  5 --seed 78 &
pids[19]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+mcdrop+oursv1 --gpu  0 --seed 79 &
pids[20]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+mcdrop+oursv1 --gpu  0 --seed 77 &
pids[21]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+mcdrop+oursv1 --gpu  3 --seed 78 &
pids[22]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+mcdrop+oursv1 --gpu  2 --seed 79 &
pids[23]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+mcdrop+oursv1 --gpu  2 --seed 77 &
pids[24]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+mcdrop+oursv1 --gpu  5 --seed 78 &
pids[25]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+mcdrop+oursv1 --gpu  0 --seed 79 &
pids[26]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+mcdrop+oursv1 --gpu  5 --seed 77 &
pids[27]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+mcdrop+oursv1 --gpu  5 --seed 78 &
pids[28]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+mcdrop+oursv1 --gpu  3 --seed 79 &
pids[29]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+mcdrop+oursv1 --gpu  5 --seed 77 &
pids[30]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+mcdrop+oursv1 --gpu  0 --seed 78 &
pids[31]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+mcdrop+oursv1 --gpu  1 --seed 79 &
pids[32]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+mcdrop+oursv1 --gpu  1 --seed 77 &
pids[33]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+mcdrop+oursv1 --gpu  2 --seed 78 &
pids[34]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+mcdrop+oursv1 --gpu  2 --seed 79 &
pids[35]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ensemble+oursv1 --gpu  2 --seed 77 &
pids[36]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ensemble+oursv1 --gpu  0 --seed 78 &
pids[37]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ensemble+oursv1 --gpu  1 --seed 79 &
pids[38]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ensemble+oursv1 --gpu  0 --seed 77 &
pids[39]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ensemble+oursv1 --gpu  1 --seed 78 &
pids[40]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ensemble+oursv1 --gpu  3 --seed 79 &
pids[41]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ensemble+oursv1 --gpu  3 --seed 77 &
pids[42]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ensemble+oursv1 --gpu  2 --seed 78 &
pids[43]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ensemble+oursv1 --gpu  4 --seed 79 &
pids[44]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ensemble+oursv1 --gpu  4 --seed 77 &
pids[45]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ensemble+oursv1 --gpu  5 --seed 78 &
pids[46]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ensemble+oursv1 --gpu  0 --seed 79 &
pids[47]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ensemble+oursv1 --gpu  4 --seed 77 &
pids[48]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ensemble+oursv1 --gpu  2 --seed 78 &
pids[49]=$!
for pid in ${pids[*]}; 
do
	 wait $pid 
done
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ensemble+oursv1 --gpu  5 --seed 79 &
pids[0]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ensemble+oursv1 --gpu  3 --seed 77 &
pids[1]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ensemble+oursv1 --gpu  2 --seed 78 &
pids[2]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-68 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ensemble+oursv1 --gpu  3 --seed 79 &
pids[3]=$!
