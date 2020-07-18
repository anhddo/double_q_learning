DIRPATH=${HOME}/tmp/BreakoutDeterministic-v4/1/logs
while :
do
    if [[ -e ${DIRPATH}/1.json ]]; then
        python -m june60.plot_result --log-path ${DIRPATH}/1.json \
            --width 10 --height 10 
        scp ${DIRPATH}/*.pdf vultr:~/image/breakout
    fi
    sleep 7m
done
