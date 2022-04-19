![fuxi_logo](https://github.com/NeteaseFuxiRL/FeverBasketball/blob/master/materials/image/Fuxi_logo.png)

# Code for the paper "Exploring Unknown States with Action Balance"

If you find this code useful, please reference in your paper:
```
@article{DBLP:journals/corr/abs-2003-04518,
  author    = {Yan Song and
               Yingfeng Chen and
               Yujing Hu and
               Changjie Fan},
  title     = {Exploring Unknown States with Action Balance},
  journal   = {CoRR},
  year      = {2020}
}
```

## Usage
**Finding unknown states (Grid world):**

Run following command for one group experiment.

```shell
cd grid-experiments && mkdir logs
./run_no_ends.sh no_ends_test run1 100 100 128 1
```

**Reaching goals (Grid world):**

```shell
cd grid-experiments && mkdir logs
./run_reach_goal.sh reach_goals_test 128 1
```

**Atari:**

This implementation is mainly based on **[random-network-distillation](https://github.com/openai/random-network-distillation)**. The following command should train an *action balance RND* with action channel on Montezuma's Revenge.

- `--abc`: 0 or 1, whether use action balance exploration. 0 means only RND.
- `--array_action`: 0 or 1, whether use action channel.

```
python3 -u run_atari.py --env=MontezumaRevengeNoFrameskip-v4 --num_env=32 --gamma_ext 0.999 --abc=1 --seed=0 --array_action=1 --logdir /tmp/action_balance_tmp_run
```

If you have any question, please contact yansong1024@gmail.com.
