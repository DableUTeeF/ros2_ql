# QtableV1

## --FiLM_dev--- ##

Copy&Run command

Example:

```terminal
python3 learning_node.py --n_actions_enable 3 --GOAL_POSITION .03  1.9777  .0
```

## Args Options

|args |type|description|
|-----|--|---------|
--log_file_dir| str | '/Data/Log_learning_CUSTOM'(default)
--Q_source_dir| str | '/Data/Log_learning_CUSTOM/Qtable.csv'(default)
--max_episodes| int | MAX_EPISODES = 10 (default)
--max_step_per_episodes| int| MAX_STEPS_PER_EPISODE = 500 (default)
--max_episodes_before_save| int|MAX_EPISODES_BEFORE_SAVE = 5 (default)
--exploration_func| int|1:Softmax(default), 2:Epsilon greedy
--resume | bool| continue learning with same Qtable True | False
--n_actions_enable| int | number of action that enable(3-8) --> 0 : forward, 1 : left, 2 : right, 3 : superForward, 4 : backward, 5 : stop, 6 : CW, 7 : CCW
--radiaus_reduce_rate| float  | RADIUS_REDUCE_RATE = .5 (default)
--reward_threshold| int | REWARD_THRESHOLD =  -200 (default)
--GOAL_POSITION| float | GOAL_POSITION = (.03, 1.9777, .0)  (default)
--GOAL_RADIUS| float | GOAL_RADIUS = .06 (default)
