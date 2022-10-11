CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ts+ours --gpu  0 --seed 77 &
pids[0]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ts+ours --gpu  1 --seed 78 &
pids[1]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ts+ours --gpu  2 --seed 79 &
pids[2]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ts+ours --gpu  3 --seed 77 &
pids[3]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ts+ours --gpu  4 --seed 78 &
pids[4]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ts+ours --gpu  5 --seed 79 &
pids[5]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ts+ours --gpu  0 --seed 77 &
pids[6]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ts+ours --gpu  1 --seed 78 &
pids[7]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ts+ours --gpu  2 --seed 79 &
pids[8]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ts+ours --gpu  3 --seed 77 &
pids[9]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ts+ours --gpu  4 --seed 78 &
pids[10]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ts+ours --gpu  5 --seed 79 &
pids[11]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ts+ours --gpu  0 --seed 77 &
pids[12]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ts+ours --gpu  1 --seed 78 &
pids[13]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ts+ours --gpu  2 --seed 79 &
pids[14]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ts+ours --gpu  3 --seed 77 &
pids[15]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ts+ours --gpu  4 --seed 78 &
pids[16]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ts+ours --gpu  5 --seed 79 &
pids[17]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+mcdrop+ours --gpu  0 --seed 77 &
pids[18]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+mcdrop+ours --gpu  1 --seed 78 &
pids[19]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+mcdrop+ours --gpu  2 --seed 79 &
pids[20]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+mcdrop+ours --gpu  3 --seed 77 &
pids[21]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+mcdrop+ours --gpu  4 --seed 78 &
pids[22]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+mcdrop+ours --gpu  5 --seed 79 &
pids[23]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+mcdrop+ours --gpu  0 --seed 77 &
pids[24]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+mcdrop+ours --gpu  1 --seed 78 &
pids[25]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+mcdrop+ours --gpu  2 --seed 79 &
pids[26]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+mcdrop+ours --gpu  3 --seed 77 &
pids[27]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+mcdrop+ours --gpu  4 --seed 78 &
pids[28]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+mcdrop+ours --gpu  5 --seed 79 &
pids[29]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+mcdrop+ours --gpu  0 --seed 77 &
pids[30]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+mcdrop+ours --gpu  1 --seed 78 &
pids[31]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+mcdrop+ours --gpu  2 --seed 79 &
pids[32]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+mcdrop+ours --gpu  3 --seed 77 &
pids[33]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+mcdrop+ours --gpu  4 --seed 78 &
pids[34]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+mcdrop+ours --gpu  5 --seed 79 &
pids[35]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ensemble+ours --gpu  0 --seed 77 &
pids[36]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ensemble+ours --gpu  1 --seed 78 &
pids[37]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+ensemble+ours --gpu  2 --seed 79 &
pids[38]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ensemble+ours --gpu  3 --seed 77 &
pids[39]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ensemble+ours --gpu  4 --seed 78 &
pids[40]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+ensemble+ours --gpu  5 --seed 79 &
pids[41]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ensemble+ours --gpu  0 --seed 77 &
pids[42]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ensemble+ours --gpu  1 --seed 78 &
pids[43]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+ensemble+ours --gpu  2 --seed 79 &
pids[44]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ensemble+ours --gpu  3 --seed 77 &
pids[45]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ensemble+ours --gpu  4 --seed 78 &
pids[46]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+ensemble+ours --gpu  5 --seed 79 &
pids[47]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ensemble+ours --gpu  0 --seed 77 &
pids[48]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ensemble+ours --gpu  1 --seed 78 &
pids[49]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+ensemble+ours --gpu  2 --seed 79 &
pids[50]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ensemble+ours --gpu  3 --seed 77 &
pids[51]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ensemble+ours --gpu  4 --seed 78 &
pids[52]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+ensemble+ours --gpu  5 --seed 79 &
pids[53]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+lula+ours --gpu  0 --seed 77 &
pids[54]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+lula+ours --gpu  1 --seed 78 &
pids[55]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+lula+ours --gpu  2 --seed 79 &
pids[56]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+lula+ours --gpu  3 --seed 77 &
pids[57]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+lula+ours --gpu  4 --seed 78 &
pids[58]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+lula+ours --gpu  5 --seed 79 &
pids[59]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+lula+ours --gpu  0 --seed 77 &
pids[60]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+lula+ours --gpu  1 --seed 78 &
pids[61]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+lula+ours --gpu  2 --seed 79 &
pids[62]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+lula+ours --gpu  3 --seed 77 &
pids[63]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+lula+ours --gpu  4 --seed 78 &
pids[64]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+lula+ours --gpu  5 --seed 79 &
pids[65]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+lula+ours --gpu  0 --seed 77 &
pids[66]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+lula+ours --gpu  1 --seed 78 &
pids[67]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+lula+ours --gpu  2 --seed 79 &
pids[68]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+lula+ours --gpu  3 --seed 77 &
pids[69]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+lula+ours --gpu  4 --seed 78 &
pids[70]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+lula+ours --gpu  5 --seed 79 &
pids[71]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method cskd+ours --gpu  0 --seed 77 &
pids[72]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method cskd+ours --gpu  1 --seed 77 &
pids[73]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method cskd+ours --gpu  2 --seed 78 &
pids[74]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method cskd+ours --gpu  3 --seed 79 &
pids[75]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method cskd+ours --gpu  4 --seed 77 &
pids[76]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method cskd+ours --gpu  5 --seed 78 &
pids[77]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method cskd+ours --gpu  0 --seed 79 &
pids[78]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method cskd+ours --gpu  1 --seed 77 &
pids[79]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method cskd+ours --gpu  2 --seed 78 &
pids[80]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method cskd+ours --gpu  3 --seed 79 &
pids[81]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method focal+ours --gpu  4 --seed 77 &
pids[82]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method focal+ours --gpu  5 --seed 78 &
pids[83]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method focal+ours --gpu  0 --seed 79 &
pids[84]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method focal+ours --gpu  1 --seed 77 &
pids[85]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method focal+ours --gpu  2 --seed 78 &
pids[86]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method focal+ours --gpu  3 --seed 79 &
pids[87]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method focal+ours --gpu  4 --seed 77 &
pids[88]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method focal+ours --gpu  5 --seed 78 &
pids[89]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method focal+ours --gpu  0 --seed 79 &
pids[90]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method bm+ours --gpu  1 --seed 77 &
pids[91]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method bm+ours --gpu  2 --seed 78 &
pids[92]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method bm+ours --gpu  3 --seed 79 &
pids[93]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method bm+ours --gpu  4 --seed 77 &
pids[94]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method bm+ours --gpu  5 --seed 78 &
pids[95]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method bm+ours --gpu  0 --seed 79 &
pids[96]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method bm+ours --gpu  1 --seed 77 &
pids[97]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method bm+ours --gpu  2 --seed 78 &
pids[98]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method bm+ours --gpu  3 --seed 79 &
pids[99]=$!
for pid in ${pids[*]}; 
do
	 wait $pid 
done
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+lula+ours --gpu  4 --seed 77 &
pids[0]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+lula+ours --gpu  5 --seed 78 &
pids[1]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method raw+lula+ours --gpu  0 --seed 79 &
pids[2]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+lula+ours --gpu  1 --seed 77 &
pids[3]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+lula+ours --gpu  2 --seed 78 &
pids[4]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.4 --method raw+lula+ours --gpu  3 --seed 79 &
pids[5]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+lula+ours --gpu  4 --seed 77 &
pids[6]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+lula+ours --gpu  5 --seed 78 &
pids[7]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.6 --method raw+lula+ours --gpu  0 --seed 79 &
pids[8]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+lula+ours --gpu  1 --seed 77 &
pids[9]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+lula+ours --gpu  2 --seed 78 &
pids[10]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method raw+lula+ours --gpu  3 --seed 79 &
pids[11]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+lula+ours --gpu  4 --seed 77 &
pids[12]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+lula+ours --gpu  5 --seed 78 &
pids[13]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method raw+lula+ours --gpu  0 --seed 79 &
pids[14]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+lula+ours --gpu  1 --seed 77 &
pids[15]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+lula+ours --gpu  2 --seed 78 &
pids[16]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method raw+lula+ours --gpu  3 --seed 79 &
pids[17]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method gp+ours --gpu  4 --seed 77 &
pids[18]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type rcn --noise_strength 0.2 --method gp+ours --gpu  5 --seed 78 &
pids[19]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method gp+ours --gpu  0 --seed 77 &
pids[20]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method gp+ours --gpu  1 --seed 78 &
pids[21]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.2 --method gp+ours --gpu  2 --seed 79 &
pids[22]=$!
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method gp+ours --gpu  3 --seed 77 &
pids[23]=$!
CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method gp+ours --gpu  4 --seed 78 &
pids[24]=$!
CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.4 --method gp+ours --gpu  5 --seed 79 &
pids[25]=$!
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method gp+ours --gpu  0 --seed 77 &
pids[26]=$!
CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method gp+ours --gpu  1 --seed 78 &
pids[27]=$!
CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=0-40 python -W ignore run_calibration.py --dataset mnist --noise_type linear --noise_strength 0.6 --method gp+ours --gpu  2 --seed 79 &
pids[28]=$!
