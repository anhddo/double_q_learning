#python -m rl.optimistic_vi.main --env CartPole-v1 --std-vi --tmp-dir tmp --n-run 25 --buffer 10000 
#python -m rl.optimistic_vi.main --env CartPole-v1 --optimistic --tmp-dir tmp --n-run 25 --buffer 10000 
#python -m rl.optimistic_vi.main --env CartPole-v1 --std-vi --tmp-dir tmp --n-run 25 --buffer 10000 
python -m rl.optimistic_vi.main --env CartPole-v1 --egreedy --tmp-dir tmp --n-run 25 --buffer 10000 
