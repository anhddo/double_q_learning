#!/bin/bash
# Ex: bash atari BreakoutDeterministic-v4
python -m june60.deep_rl.run_atari --env $1 --tmp-dir ~/tmp --dqn \
    --vi --mse --adam --lr 0.0001 
