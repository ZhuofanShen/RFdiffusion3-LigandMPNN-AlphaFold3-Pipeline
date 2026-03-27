# RFdiffusion3-LigandMPNN-AlphaFold3-Pipeline
This repository provides an automated workflow that sequentially runs RFdiffusion3, LigandMPNN, and AlphaFold3, followed by result filtering, for de novo ligand binder design, using PLP-dependent pyruvic acid transaminase design as an example application.

```
git clone https://github.com/ZhuofanShen/RFdiffusion3-LigandMPNN-AlphaFold3-Pipeline
cd RFdiffusion3-LigandMPNN-AlphaFold3-Pipeline
```

## RFdiffusion3

```
python scripts/run_rfdiffusion3_wrapper.py \
  -j /home/szf/rfd3-ligandmpnn-af3-pipeline/inputs/rfd3_json_templates/Cys_PLP_options_3.json \
  -i /home/szf/rfd3-ligandmpnn-af3-pipeline/inputs/Cys_PLP \
  -o /home/szf/rfd3-ligandmpnn-af3-pipeline/outputs/Cys_PLP_options_3 \
  -b 10 -n 10 --max_workers 8
```

Run the following command for rotamers 0-5 in separate tmux sessions/slurm jobs.
```
conda activate aifold

i=0
CUDA_VISIBLE_DEVICES=${i}
rfd3 design inputs=/home/szf/rfd3-ligandmpnn-af3-pipeline/inputs/Cys_PLP/Cys_PLP_rot_${i}_Cys_PLP_options_3.json out_dir=/home/szf/rfd3-ligandmpnn-af3-pipeline/outputs/Cys_PLP_options_3 skip_existing=False prevalidate_inputs=True n_batches=10 diffusion_batch_size=10
```

## LigandMPNN

Run the following command for rotamers 0-5 in separate tmux sessions/slurm jobs.
```
python scripts/run_ligandmpnn_from_rfd3.py -w "Cys_PLP_rot_${i}_Cys_PLP_options_3_*_model_*.cif.gz" -od "/home/szf/rfd3-ligandmpnn-af3-pipeline/outputs/Cys_PLP_options_3" -cat A403 -s 10 --gpu ${i}
```

## AlphaFold3

Run the following command for rotamers 0-5 in separate tmux sessions/slurm jobs.
```
python scripts/run_alphafold3_from_ligandmpnn.py -j "inputs/Cys_PLP_rot_"${i}"_options_3_af3_jsons" -rfd3o "outputs/Cys_PLP_options_3" -w "Cys_PLP_rot_"${i}"_Cys_PLP_options_3_*_*_model_*" -l inputs/cifs/PLP_pyruvic_acid_quinonoid.cif -b A403,SG B1,C5A --gpu 7

unset CUDA_VISIBLE_DEVICES

docker run -it --gpus device=${i} --volume .:/work --volume /opt/alphafold3_weights:/root/models --volume /home/public_databases:/root/public_databases alphafold3 python run_alphafold.py --input_dir /work/inputs/Cys_PLP_rot_${i}_options_3_af3_jsons --model_dir /root/models --norun_data_pipeline --output_dir /work/outputs/Cys_PLP_options_3_AlphaFold3 --num_recycles 1 --num_diffusion_samples 1
```

## Filter designs according to pLDDT, iPAE, and ipTM confidence scores

Run the following command for rotamers 0-5 in separate tmux sessions/slurm jobs.
```
python scripts/collect_af3_results_two_state_best_seq.py -af3o "outputs/Cys_PLP_options_3_AlphaFold3" -w "Cys_PLP_rot_${i}_*_model_*" -o "outputs/Cys_PLP_rot_${i}_options_3_AlphaFold3_analysis"
```
