for slurm_script in $(ls scripts/prune_llama3_d_*.sh); do
    echo "sbatch $slurm_script"
    sbatch $slurm_script
done