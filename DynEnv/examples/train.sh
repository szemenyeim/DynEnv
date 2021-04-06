tmux new-session -d -s RL
#for seed in 5646 83465 97 4162 35710 4380
for seed in 42 658 94846 341
do
    NOTE="seed-${seed}-"
    ARGS=""

    for arg in "$@"
    do
         if [ "$arg" == "--rcm" ]
        then
            ARGS+=" --use-rcm True"
            NOTE+="RCM-"
        else
            NOTE+="ICM-"
        fi

        if [ "$arg" == "--partial" ]
        then
            ARGS+=" --observationType PARTIAL"
            NOTE+="PartialObs-"
        elif [ "$arg" == "--full" ]
        then
            ARGS+=" --observationType FULL --noiseMagnitude 0"
            NOTE+="FullObs-"
        fi
        
        if [ "$arg" == "--reconstruction" ]
        then
            ARGS+=" --use-reconstruction True --recon-pretrained True"
            NOTE+="Recon-"
        fi
        

        if [ "$arg" == "--ppo" ]
        then
            ARGS+=" --use-ppo True"
            NOTE+="PPO-"
        fi

        if [ "$arg" == "--longterm" ]
        then
            ARGS+=" --long-horizon-coeff 1e-3"
            NOTE+="LongTermPred"
        fi



    done
   # echo "$seed"
   # echo "$NOTE"
   # echo "$ARGS"
   # echo "python3 -W ignore main.py --seed $seed --note $NOTE $ARGS"
   CMD="python3 -W ignore main.py --seed ${seed} --note ${NOTE} ${ARGS}"
   echo "$CMD"
   tmux new-window -t RL:$seed
   tmux send-keys -t RL:$seed "$CMD" C-m
   sleep 5
done
