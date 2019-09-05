#!bin/bash

# set -xe

if [ $# -lt 3 ]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer speed|mem|maxbs sp|mp /ssd3/benchmark_results/cwh/logs"
    exit
fi

#打开后速度变快
export FLAGS_cudnn_exhaustive_search=1

#显存占用减少，不影响性能
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=1 
export FLAGS_conv_workspace_size_limit=256

task="$1"
index="$2"
run_mode="$3"
run_log_path=${4:-$(pwd)}
model_name="seq2seq"
min=${5:-1}
max=${6:-2048}

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}
batch_size_=1000
log_file_=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}_${run_mode}

train(){
    batch_size=${1:-$batch_size_}
    batch_size=`expr ${batch_size} \* ${num_gpu_devices}`
    log_file=${log_file_}_${batch_size}

    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd=" --src_lang en --tar_lang vi \
        --attention True \
        --num_layers 2 \
        --hidden_size 512 \
        --src_vocab_size 17191 \
        --tar_vocab_size 7709 \
        --batch_size ${batch_size} \
        --dropout 0.2 \
        --init_scale  0.1 \
        --max_grad_norm 5.0 \
        --train_data_prefix data/en-vi/train \
        --eval_data_prefix data/en-vi/tst2012 \
        --test_data_prefix data/en-vi/tst2013 \
        --vocab_prefix data/en-vi/vocab \
        --use_gpu True \
	--max_epoch 1"

    case ${run_mode} in
    sp) 
        train_cmd="python3 -u train.py "${train_cmd} 
	echo $train_cmd
        ;;
    mp) 
        train_cmd="python3 -m paddle.distributed.launch --log_dir=./my_log --selected_gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" 
        ;;
    *)  
        echo "choose run_mode: sp or mp" 
        exit 1 
        ;;
    esac

    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    #sleep 600
    #kill -9 $train_pid
    echo $train_pid
    wait $train_pid
    echo '111'

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi

    error_string="Cannot malloc"
    if [ `grep -c "${error_string}" ${log_file}` -eq 0 ]; then
      return 0
    else
      return 1
    fi
}

analysis_times(){
    skip_step=$1
    awk 'BEGIN{count=0}/Batch_time_cost:/{
      count_fields=NF;
      step_times[count]=$count_fields;
      count+=1;
    }END{
      print "\n================ Benchmark Result ================"
      print "total_step:", count
      print "batch_size:", "'${batch_size}'"
      if(count>1){
        step_latency=0
        step_latency_without_step0_avg=0
        step_latency_without_step0_min=step_times['${skip_step}']
        step_latency_without_step0_max=step_times['${skip_step}']
        for(i=0;i<count;++i){
          step_latency+=step_times[i];
          if(i>='${skip_step}'){
            step_latency_without_step0_avg+=step_times[i];
            if(step_times[i]<step_latency_without_step0_min){
              step_latency_without_step0_min=step_times[i];
            }
            if(step_times[i]>step_latency_without_step0_max){
              step_latency_without_step0_max=step_times[i];
            }
          }
        }
        step_latency/=count;
        step_latency_without_step0_avg/=(count-'${skip_step}')
        printf("average latency (origin result):\n")
        printf("\tAvg: %.3f s/step\n", step_latency)
        printf("\tFPS: %.3f images/s\n", "'${batch_size}'"/step_latency)
        printf("average latency (skip '${skip_step}' steps):\n")
        printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
        printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
        printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
        printf("\tFPS: %.3f images/s\n", '${batch_size}'/step_latency_without_step0_avg)
        printf("\n")
      }
    }' ${log_parse_file}
}

echo "Benchmark for $task"

if [ $index = "mem" ]
then
    echo 'abc'
    echo $task
    # 若测试最大batchsize，FLAGS_fraction_of_gpu_memory_to_use=1
    export FLAGS_fraction_of_gpu_memory_to_use=0
    gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > ${log_file_}_gpu_use 2>&1 &
    #nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    echo $gpu_memory_pid
    $task
    kill $gpu_memory_pid
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' ${log_file_}_gpu_use
elif [ $index = 'maxbs' ]
then
    export FLAGS_fraction_of_gpu_memory_to_use=1
    while [ $min -lt $max ]; do
     	current=`expr '(' "${min}" + "${max}" + 1 ')' / 2`
    	    echo "Try batchsize=${current}"
    	    if train ${current}; then
    		min=${current}
    	    else
    		max=`expr ${current} - 1`
    	    fi
    done

else 
    job_bt=`date '+%Y%m%d%H%M%S'`
    $task
    job_et=`date '+%Y%m%d%H%M%S'`
    hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
    # monquery -n $hostname -i GPU_AVERAGE_UTILIZATION -s $job_bt -e $job_et -d 60 > gpu_avg_utilization
    # monquery -n $hostname -i CPU_USER -s $job_bt -e $job_et -d 60 > cpu_use
    cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
    gpu_num=$(nvidia-smi -L|wc -l)
    # awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("avg_gpu_use=%.2f\n" ,avg*'${gpu_num}')}' gpu_avg_utilization
    # awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("avg_cpu_use=%.2f\n" ,avg*'${cpu_num}')}' cpu_use
    
    if [ ${task} = "train" ]
    then
      analysis_times 3 
    else
      echo "no infer cmd"
    fi
fi

