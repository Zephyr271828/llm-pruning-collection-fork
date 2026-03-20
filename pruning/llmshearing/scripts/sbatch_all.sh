for slurm_script in $(ls scripts/prune_llama3_d_3456*.sh); do
    echo "sbatch $slurm_script"
    sbatch $slurm_script
done