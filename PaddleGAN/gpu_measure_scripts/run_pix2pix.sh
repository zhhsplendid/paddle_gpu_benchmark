#!bin/bash
set -xe

if [ $# -lt 1 ]; then
  echo "Usage: "
  echo " CUDA_VISIBLE_DEVICES=0 bash run_pix2pix.sh mem|maxbs logs"
  exit
fi

# Configuration of Allocator and GC
export FLAGS_fraction_of_gpu_memory_to_use=1.0
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_memory_fraction_of_eager_deletion=1.0

index="$1"
run_log_path=${2:-$(pwd)}

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}

base_batch_size=156

batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
log_file=${run_log_path}/pix2pix_${index}_${num_gpu_devices}gpu.log

train() {
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

  python train.py \
    --model_net Pix2pix \
    --dataset cityscapes \
    --train_list data/cityscapes/pix2pix_train_list \
    --test_list data/cityscapes/pix2pix_test_list \
    --crop_type Random \
    --dropout True \
    --gan_mode vanilla \
    --batch_size ${base_batch_size} \
    --epoch 2 \
    --image_size 286 \
    --crop_size 256 \
    > ${log_file} 2>&1 &

  train_pid=$!
  sleep 120
  # kill -9 $train_pid
  kill -9 `ps -ef|grep python |awk '{print $2}'` || true
}

if [ $index = "mem" ]
then
  export FLAGS_fraction_of_gpu_memory_to_use=0
  base_batch_size=1
  gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
  nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 1000 > gpu_use.log 2>&1 &
  train
  gpu_memory_pid=$!
  kill -9 $gpu_memory_pid || true
  awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
else
  train
  error_string="Please shrink FLAGS_fraction_of_gpu_memory_to_use"
  if [ `grep -c "${error_string}" ${log_file}` -eq 0 ]; then
    echo "maxbs is ${batch_size}"
  else
    echo "maxbs running error"
  fi
fi
