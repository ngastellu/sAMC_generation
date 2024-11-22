The code in this repository can be used to generate amorphous monolayer carbon (AMC) fragments using the
PixelCNN implementation of the morphological autoregression protocol (MAP).

This code was used to generate the sAMC-q400 and sAMC-300 ensembles of structures in the 
manuscript titled "Disentangling the morphology-conductance relationship in amorphous graphene using deep learning and percolation theory" which has been submitted for publication.

The scripts in this repository were ran on the Narval HPC cluster operated by Calcul Québec, which 
uses the SLURM scheduler. The script `rungenerateog.sh` submits a generation to Narval's compute 
nodes by calling the `model_save_load_generateog.py` Python script, which loads the weights of 
PixelCNN model and generates an AMC structure. 

To generate a structure a structure from the sAMC-q400 ensemble, run the following command:
`sbatch rungenerateog.sh 0.6`; this sets the softmax temperature parameter to 0.6 (see Methods
section of manuscript for details).

For a structure from the sAMC-300 ensemble, run `sbatch rungenerateog.sh 0.5`.

Note that if you plan on using this code on a computing cluster which uses another scheduler,
you will have to modify `rungenerateog.sh`.

The file `model-checkpoint-1.pt` contains the weights of the CNN used for the generation.
Take care not to overwrite it.
