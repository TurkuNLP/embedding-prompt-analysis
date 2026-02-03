for shard in {0..9}
do
    command="sbatch -J e-$shard -o logs/e-$shard.eo embed.sbatch.sh $shard 10"
    echo $command
done
