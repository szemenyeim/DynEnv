tmux new-session -d -s RL
for seed in 42 658 94846 341 #35710 4380 5646 83465 97 4162
do
    NOTE="seed-${seed}-"
    ARGS=""

    for arg in "$@"
    do
        if [ "$arg" == "--partial" ]
        then
            ARGS+=" --observationType ObservationType.PARTIAL"
            NOTE+="PartialObs-"
        elif [ "$arg" == "--full" ]
        then
            ARGS+=" --observationType ObservationType.FULL"
            NOTE+="FullObs-"
        fi
        
        if [ "$arg" == "--reconstruction" ]
        then
            ARGS+=" --use-reconstruction True --recon-pretrained True"
            NOTE+="Recon-"
        fi
        
        
        if [ "$arg" == "--longterm" ]
        then
            ARGS+=" --long-horizon-coeff 1e-3"
            NOTE+="LongTermPred"
        else
            ARGS+=" --long-horizon-coeff 0.0"
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
   sleep 0.5
done
