NAME=$1
BATCH_NAM=$2
RUN_TIME=$3
GMAE_NUM=$4
MSF=$5
ACC=$6
SRC='0.1'

name_prefix=${NAME}_msf${MSF}_acc${ACC}_src${SRC}

# random
python3 -u run_no_learner.py --pf=random --ep=0 --acc=0 --src=${SRC} \
        --save_name logs/${name_prefix}-ep1/${BATCH_NAM} \
        --total_times=${GMAE_NUM} --run_times=${RUN_TIME} --log_run_index=-1 --msf=${MSF} > /dev/null 2>&1 &

# RND
python3 -u run_no_learner.py --pf=buffer --ep=0 --acc=0 --src=${SRC} \
        --save_name logs/${name_prefix}-src/${BATCH_NAM} \
        --total_times=${GMAE_NUM} --run_times=${RUN_TIME} --log_run_index=-1 --msf=${MSF} > /dev/null 2>&1 &

# action balance RND
python3 -u run_no_learner.py --pf=buffer --ep=0 --acc=${ACC} --src=${SRC} \
        --save_name logs/${name_prefix}-acc_src/${BATCH_NAM} \
        --total_times=${GMAE_NUM} --run_times=${RUN_TIME} --log_run_index=-1 --msf=${MSF} > /dev/null 2>&1 &

# action balance
python3 -u run_no_learner.py --pf=buffer --ep=0 --acc=${ACC} --src=0 \
        --save_name logs/${name_prefix}-acc/${BATCH_NAM} \
        --total_times=${GMAE_NUM} --run_times=${RUN_TIME} --log_run_index=-1 --msf=${MSF} > /dev/null 2>&1 &

wait

echo "All done."
