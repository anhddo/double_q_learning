# bash plot.sh BreakoutDeterministic-v4 1
DIRPATH=${HOME}/tmp/$1/$2/logs
JSON=${DIRPATH}/1.json
while :
do
    if [[ -e ${JSON} ]]; then
        python -m june60.plot_result --log-path ${JSON} \
            --width 10 --height 10 
        ssh vultr "mkdir -p ~/server/$1/$2"
        scp ${DIRPATH}/*.pdf vultr:~/server/$1/$2
    fi
    sleep 3m
done
