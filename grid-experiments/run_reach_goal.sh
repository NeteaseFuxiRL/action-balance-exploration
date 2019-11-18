NAME=$1
MSF=$2
ACC=$3
SRC='0.1'

name_prefix=${NAME}_msf${MSF}_acc${ACC}_src${SRC}

# random
python3 -u run_no_learner.py --pf=random --ep=0 --acc=0 --src=${SRC} --msf=${MSF} --ends="all" \
        > logs/${name_prefix}-ep1_ends.log  2>&1 &
# RND
python3 -u run_no_learner.py --pf=buffer --ep=0 --acc=0 --src=${SRC} --msf=${MSF} --ends="all" \
        > logs/${name_prefix}-src_ends.log 2>&1 &
# action balance RND
python3 -u run_no_learner.py --pf=buffer --ep=0 --acc=${ACC} --src=${SRC} --msf=${MSF} --ends="all" \
        > logs/${name_prefix}-acc_src_ends.log 2>&1 &

wait

echo "All done."
