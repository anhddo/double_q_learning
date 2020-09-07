python -m rl.deep_rl.classical.run --tmp-dir tmp --env CartPole-v1 --ddqn --vi --huber --rms --n-run 10 --learn-start 32 --latent-buffer-size 50000 --no-explore
python -m rl.deep_rl.classical.run --tmp-dir tmp --env CartPole-v1 --ddqn --vi --huber --rms --n-run 10 --learn-start 32 --latent-buffer-size 50000 --optimistic --beta 5
python -m rl.deep_rl.classical.run --tmp-dir tmp --env CartPole-v1 --ddqn --vi --huber --rms --n-run 10 --learn-start 32 --latent-buffer-size 50000 --epsilon-greedy
