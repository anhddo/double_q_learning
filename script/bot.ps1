#python -m june60.optimistic_vi.main --env Acrobot-v1 --std-vi --tmp-dir tmp --n-run 25 --buffer 10000 
python -m june60.optimistic_vi.main --env Acrobot-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 5000 --fourier-order 2 --beta 0.5
python -m june60.optimistic_vi.main --env Acrobot-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 5000 --fourier-order 2 --beta 1
python -m june60.optimistic_vi.main --env Acrobot-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 5000 --fourier-order 2 --beta 5
python -m june60.optimistic_vi.main --env Acrobot-v1 --optimistic --tmp-dir tmp --n-run 50 --buffer 5000 --fourier-order 2 --beta 10
