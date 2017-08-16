#!/bin/bash                                                                                                                              
#                                                                                                                                        
# Run the deepcis model                                                                                                                  
#                                                                                                                                        
# author: avsec                                                                                                                          
############################################                                                                                             

#SBATCH --mem=16G                                                                                                                        
#SBATCH -c 4                                                                                                                             
DB=SeqScorePrediction17
export PYTHONPATH=/data/nasif12/home_if12/nguythiy/workspace/psi_cnn/donor/hyperopt

                                                                                                                                        
hyperopt-mongo-worker \
    --mongo=ouga03:1234/$DB \
    --poll-interval=1 \
    --reserve-timeout=3600

