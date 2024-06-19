for seed in 0 1 2 3 4;
do
    cuda_id=`expr $seed + 5`
    CUDA_VISIBLE_DEVICES=$cuda_id python main.py --optimizer dadaptsgd --seed $seed --lr 1.0 &
done
wait

for seed in 0 1 2 3 4;
do
    cuda_id=`expr $seed + 5`
    CUDA_VISIBLE_DEVICES=$cuda_id python main.py --optimizer dadaptsgd --seed $seed --lr 0.1 &
done
wait