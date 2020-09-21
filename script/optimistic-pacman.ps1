python -m rl.deep_rl.atari.run_atari --env MsPacmanDeterministic-v4 --tmp-dir tmp --ddqn --vi --huber --rms --optimistic --beta 10
#Test optimistic
#python -m rl.deep_rl.atari.run_atari --env MsPacmanDeterministic-v4 --tmp-dir tmp --ddqn --vi --huber --rms --buffer 100 --optimistic --learn-start 50 --eval-step 10 --beta 10 --epoch-step 100
#python -m rl.deep_rl.atari.run_atari --env MsPacmanDeterministic-v4 --tmp-dir tmp --ddqn --vi --huber --rms --learn-start 50 --optimistic --epoch-step 100 --eval-step 100
