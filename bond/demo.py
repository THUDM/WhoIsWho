from training.autotrain_bond import BONDTrainer
from training.autotrain_bond_ensemble import ESBTrainer
from dataset.preprocess_SND import dump_name_pubs, dump_features_relations_to_file, build_graph
from params import set_params

args = set_params()

def pipeline(model):
    # Module-1: Data Loading
    dump_name_pubs()
    dump_features_relations_to_file()
    build_graph()

    # Modules-2: Feature Creation & Module-3: Model Construction
    if model == 'bond':
        trainer = BONDTrainer()
        trainer.fit(datatype=args.mode)
    elif model == 'bond+':
        trainer = ESBTrainer()
        trainer.fit(datatype=args.mode)

    # Modules-4: Evaluation
    # Please uppload your result to http://whoiswho.biendata.xyz/#/

if __name__ == "__main__":
    pipeline(model="bond")