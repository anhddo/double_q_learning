
#!/bin/bash
# Ex: bash atari BreakoutDeterministic-v4
python -m june60.deep_rl.run_atari --env $1 --tmp-dir ~/tmp --ddqn \
    --vi --huber --rms --init-exploration 1 --final-exploration 0.01 \
    --eval-exploration 0.001 --final-exploration-step 1000000

