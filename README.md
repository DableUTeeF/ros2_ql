# QtableV1

## --welocomeToV2--- ##

Copy&Run command

Example:

```terminal
python3 learning_node.py --n_actions_enable 3 --GOAL_POSITION .03  1.9777  .0
```

## Args Options

|args |type|description|
|-----|--|---------|
--log_file_dir| str | '/Data/Log_learning'(default)
--Q_source_dir| str | '/Data/Log_learning/Qtable.csv'(default)
--max_episodes| int | MAX_EPISODES = 10 (default)
--max_step_per_episodes| int| MAX_STEPS_PER_EPISODE = 500 (default)
--max_episodes_before_save| int|MAX_EPISODES_BEFORE_SAVE = 5 (default)
--exploration_func| int|1:Softmax(default), 2:Epsilon greedy
--get_best | bool| save Qtable_best  when minimize rel_time --> True | False
--resume | bool| continue learning with same Qtable True | False
--n_actions_enable| int | number of action that enable(3-5  default:4) --> 0:forward, 1:CW, 2:CCW, 3:stop, 4:superForward
--radiaus_reduce_rate| float  | RADIUS_REDUCE_RATE = .5 (default)
--reward_threshold| int | REWARD_THRESHOLD =  -200 (default)
--GOAL_POSITION| float | GOAL_POSITION = (.03, 1.9777, .0)  (default)
--GOAL_RADIUS| float | GOAL_RADIUS = .06 (default)
