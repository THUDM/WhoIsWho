import argparse
import copy
import time
from whoiswho.dataset import LoadData,processdata_RND,processdata_SND
from whoiswho.featureGenerator.rndFeature import AdhocFeatures , OagbertFeatures ,GraphFeatures
from whoiswho.training import RNDTrainer,SNDTrainer
from whoiswho.evaluation import evaluate
from whoiswho.utils import load_json
from whoiswho.config  import *
from whoiswho import logger

def download_data(name,task):
    for type in ['train','valid','test']:
        try:
            _,version = LoadData(name = name , task = task, type = type)
        except:
            logger.error(f"Error in load_task_data for name: {name} task:{task} type:{type}")


def generate_feature(version: dict,feature_list: list,type_list: list):
    '''
        Used for RND task to generate features
    '''
    for feature in feature_list:
        assert feature  in ['adhoc','oagbert','graph'] , f'feature not supported: {feature}'
        for type in type_list:
            assert type in ['train','valid','test'], f'type not supported: {type}'
            present_version = copy.deepcopy(version)
            present_version["type"] = type
            if feature == 'adhoc':
                adhoc_features = AdhocFeatures(present_version)
                adhoc_features.get_hand_feature()

            if feature == 'oagbert':
                oagbert_features = OagbertFeatures(present_version)
                oagbert_features.get_oagbert_feature()


            if feature == 'graph':
                if type == 'train':
                    adhoc_features = AdhocFeatures(present_version, graph_data=True)  # whoiswhograph hand feature
                    adhoc_features.get_hand_feature()

                graph_features = GraphFeatures(present_version)
                graph_features.get_graph_feature()




def pipeline(name: str , task: str ,type_list: list, feature_list: list):
    '''
        According to the 'task'，disambiguate the dataset corresponding to dataset 'name'.
    '''
    version = {"name": name, "task": task}
    if task == 'RND':
        # Module-1: Data Loading
        download_data(name,task)
        # Partition the training set into unassigned papers and candidate authors
        processdata_RND(version=version)

        # Modules-2: Feature Creation
        generate_feature(version,
                         type_list = type_list,
                         feature_list = feature_list)

        # Module-3: Model Construction
        trainer = RNDTrainer(version,simplified=False,graph_data=True) #You can save time by using a simplified model
        cell_model_list = trainer.fit()
        trainer.predict(whole_cell_model_list=cell_model_list,datatype='valid')

        # Modules-4: Evaluation
        # Please uppload your result to http://whoiswho.biendata.xyz/#/

    if task == 'SND':
        # Module-1: Data Loading
        download_data(name, task)
        processdata_SND(version=version)

        # Modules-2: Feature Creation & Module-3: Model Construction
        trainer = SNDTrainer(version)
        trainer.fit(datatype='valid')

        # Modules-4: Evaluation
        # Please uppload your result to http://whoiswho.biendata.xyz/#/





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WhoIsWho Toolkit')
    parser.add_argument('-n','--name', type=str, default="v3")
    parser.add_argument('--task', type=str.upper, default="RND",
                        choices=['RND', 'SND'])
    parser.add_argument('--type', dest='type_list',nargs='+')
    parser.add_argument('--feature', dest='feature_list', nargs='+')
    args = parser.parse_args()

    name = args.name
    task = args.task
    type_list = args.type_list
    feature_list = args.feature_list
    assert task == 'RND' or task == 'SND'

    pipeline(name=name,task=task,type_list=type_list,feature_list=feature_list)
    # pipeline(name='NA_Demo',task='SND')

