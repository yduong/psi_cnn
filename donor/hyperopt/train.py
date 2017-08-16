from hyperopt import fmin, tpe, hp
from concise.hyopt import CompileFN, CMongoTrials
from copy import deepcopy
import numpy as np

import sys
sys.path.append('../../helper')
import common as cm


from model import sum_r2_score, testr2_score

KILL_TIMEOUT = 60 * 2400  # 600 minutes

DB_NAME = "SeqScorePrediction17"
# --------------------------------------------
exp_name = "cdd_morelayer_params"
#print_exp(exp_name)
# -----


    
fn = CompileFN(DB_NAME, exp_name, add_eval_metrics=[testr2_score],
                            loss_metric="testr2_score",  # val_loss
                            loss_metric_mode="max",
                       data_fn=cm.get_data,
                       model_fn=cm.get_model)



hyper_params = {
        "data": {"fold": 0
             },
    "model": {
              'filters': hp.choice("filters", [32,64,128]),
              'motif_width': hp.choice("motif_width", [7,9,11,13,15]), # use odd numbers - 11 or 15
               'use_1by1_conv': hp.choice('use_1by1_conv',[False,True]), # hp.choice(True, False(
              # regularization
              'l1': hp.loguniform("l1", np.log(1e-12), np.log(1e-6)),   
              # dense layers
              'hidden':hp.choice('hidden_layer',[None,{'n_hidden':hp.choice("n_hidden", [32,64,128]),
                      'n_units': hp.choice("n_units", [32,64,128]),
                      'dropout': hp.uniform('dropout',0,0.7),
                      'activation':  hp.choice("h_activation",["relu","sigmoid", "tanh"]),
                      'l1': hp.loguniform("h_l1", np.log(1e-12), np.log(1e-6))
                      }]),
              "task_specific":hp.choice("task_specific",[None,{"l1_1":hp.loguniform("l1_1", np.log(1e-12), np.log(1e-6)),
                      'l1_2' : hp.loguniform("l1_2", np.log(1e-12), np.log(1e-6)),
                      'l1_3' : hp.loguniform("l1_3", np.log(1e-12), np.log(1e-6)),
                      }]),
              'lr': hp.loguniform("m_lr", np.log(1e-4), np.log(1e-2))
              },
    "fit": {"epochs": 300,
            "patience": 10,
            "batch_size": 32,
            }
}


for i in range(10):
    hparams = deepcopy(hyper_params)
    hparams["data"]["fold"] = i
    c_exp_name = exp_name + "_fold_{0}".format(i)
    trials = CMongoTrials(DB_NAME, c_exp_name, kill_timeout=KILL_TIMEOUT)
    best = fmin(fn, hparams, trials=trials, algo=tpe.suggest, max_evals=100)
    print("best_parameters: " + str(i) + " :" + str(best))

    



