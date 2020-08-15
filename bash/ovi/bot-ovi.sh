#OVI
python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 1.0 --min-epsilon 0.0 --beta .5 \
    --buffer 5000 --training-step 30000 --train-freq 1
# greedy 
python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 0.1 --min-epsilon 0.1 \
    --buffer 5000 --training-step 30000 --train-freq 1
#Standard
python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --beta 0 --min-epsilon 0 --max-epsilon 1 \
    --buffer 5000 --training-step 30000 --train-freq 1
#python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 1.0 --min-epsilon 0.0 --beta .5 \
#    --buffer 5000 --training-step 30000 --train-freq 4
#python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 1.0 --min-epsilon 0.0 --beta 1 \
#    --buffer 5000 --training-step 30000 --train-freq 4
#python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 1.0 --min-epsilon 0.0 --beta 5 \
#    --buffer 5000 --training-step 30000 --train-freq 4
#python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 1.0 --min-epsilon 0.0 --beta .5 \
#    --buffer 10000 --training-step 30000 --train-freq 4
#python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 1.0 --min-epsilon 0.0 --beta 1 \
#    --buffer 10000 --training-step 30000 --train-freq 4
#python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 1.0 --min-epsilon 0.0 --beta 5 \
#    --buffer 10000 --training-step 30000 --train-freq 4
#python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 1.0 --min-epsilon 0.0 --beta .5 \
#    --buffer 15000 --training-step 30000 --train-freq 4
#python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 1.0 --min-epsilon 0.0 --beta 1 \
#    --buffer 15000 --training-step 30000 --train-freq 4
#python -m june60.optimistic_vi.main --env Acrobot-v1 --tmp-dir tmp --n-run 50 --max-epsilon 1.0 --min-epsilon 0.0 --beta 5 \
#    --buffer 15000 --training-step 30000 --train-freq 4
