python -m rl.optimistic_vi.main --env Acrobot-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 1
python -m rl.optimistic_vi.main --env Acrobot-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 5
python -m rl.optimistic_vi.main --env Acrobot-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 10
python -m rl.optimistic_vi.main --env Acrobot-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 15
python -m rl.optimistic_vi.main --env Acrobot-v1 --std-vi --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 
python -m rl.optimistic_vi.main --env Acrobot-v1 --egreedy --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 

python -m rl.optimistic_vi.main --env CartPole-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 1
python -m rl.optimistic_vi.main --env CartPole-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 5
python -m rl.optimistic_vi.main --env CartPole-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 10
python -m rl.optimistic_vi.main --env CartPole-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 15
python -m rl.optimistic_vi.main --env CartPole-v1 --std-vi --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 
python -m rl.optimistic_vi.main --env CartPole-v1 --egreedy --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 

python -m rl.optimistic_vi.main --env MountainCar-v0 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 1
python -m rl.optimistic_vi.main --env MountainCar-v0 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 5
python -m rl.optimistic_vi.main --env MountainCar-v0 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 10
python -m rl.optimistic_vi.main --env MountainCar-v0 --optimistic --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 --beta 15
python -m rl.optimistic_vi.main --env MountainCar-v0 --std-vi --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 
python -m rl.optimistic_vi.main --env MountainCar-v0 --egreedy --tmp-dir tmp --n-run 50 --buffer 25000 --fourier-order 1 
