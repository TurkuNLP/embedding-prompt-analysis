for shard in {0..9}
do
    command="sbatch -J e-e5-$shard -o logs/e-e5-$shard.eo embed-mahti.sbatch.sh $shard 10 intfloat/multilingual-e5-large-instruct multilingual-e5-large-instruct"
    echo $command
done
