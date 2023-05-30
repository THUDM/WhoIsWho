import argparse

from whoiswho.dataset import LoadData,processdata_RND,processdata_SND
from whoiswho.featureGenerator.rndFeature import AdhocFeatures , OagbertFeatures
from whoiswho.training import RNDTrainer,SNDTrainer
from whoiswho.evaluation import evaluate
from whoiswho.utils import load_json
from whoiswho.config  import *
from whoiswho import logger


def rnd_demo():
    # Module-1: Data Loading
    train, version = LoadData(name="v3", type="train", task='RND')
    valid, version = LoadData(name="v3", type="valid", task='RND')
    test, version = LoadData(name="v3", type="test", task='RND')

    # Split data into unassigned papers and candidate authors
    # Combine unassigned papers and candidate authors into train pairs.
    train, version = LoadData(name="v3", type="train", task='RND')
    processdata_RND(train, version)
    logger.info("Finish pre-process")

    # Modules-2: Feature Creation
    version = LoadData(name="v3", type="train", task='RND', just_version=True)
    adhoc_features = AdhocFeatures(version)
    adhoc_features.get_hand_feature()
    oagbert_features = OagbertFeatures(version)
    oagbert_features.get_oagbert_feature()
    logger.info("Finish Train data")

    version = LoadData(name="v3", type="valid", task='RND', just_version=True)
    adhoc_features = AdhocFeatures(version)
    adhoc_features.get_hand_feature()
    oagbert_features = OagbertFeatures(version)
    oagbert_features.get_oagbert_feature()
    logger.info("Finish Valid data")

    version = LoadData(name="v3", type="test", task='RND', just_version=True)
    adhoc_features = AdhocFeatures(version)
    adhoc_features.get_hand_feature()
    oagbert_features = OagbertFeatures(version)
    oagbert_features.get_oagbert_feature()
    logger.info("Finish Test data")

    # Module-3: Model Construction
    version = LoadData(name="v3", type="train", task='RND', just_version=True)
    trainer = RNDTrainer(version)
    cell_model_list = trainer.fit()
    logger.info("Finish Training")

    version = LoadData(name="v3", type="valid", task='RND', just_version=True)
    trainer = RNDTrainer(version)
    trainer.predict(cell_model_list=cell_model_list) #Use trained model or model path.
    logger.info("Finish Predict Valid data")

    version = LoadData(name="v3", type="test", task='RND', just_version=True)
    trainer = RNDTrainer(version)
    trainer.predict(cell_model_list=cell_model_list) #Use trained model or model path.
    logger.info("Finish Predict Test data")

    # Modules-4: Evaluation
    # Upload the results to the whoiswho competition


def snd_demo():
    # Module-1: Data Loading
    train, version = LoadData(name="v3", type="train", task='SND')
    valid, version = LoadData(name="v3", type="valid", task='SND')
    test, version = LoadData(name="v3", type="test", task='SND')
    processdata_SND(train, version)
    logger.info("Finish pre-process")

    # Modules-2: Feature Creation & Module-3: Model Construction
    version = LoadData(name="v3", type="train", task='SND',just_version=True)
    trainer = SNDTrainer(version)
    trainer.fit()
    logger.info("Finish Predict Train data")

    version = LoadData(name="v3", type="valid", task='SND',just_version=True)
    trainer = SNDTrainer(version)
    trainer.fit()
    logger.info("Finish Predict Valid data")

    version = LoadData(name="v3", type="test", task='SND',just_version=True)
    trainer = SNDTrainer(version)
    trainer.fit()
    logger.info("Finish Predict Test data")

    #Modules-4: Evaluation
    #Upload the results to the whoiswho competition




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='whole process')
    parser.add_argument('--task_type', type=str, default="rnd",
                        choices=['rnd', 'snd'])
    args = parser.parse_args()
    task_type = args.task_type
    assert task_type == 'rnd' \
           or task_type == 'snd'

    if task_type == 'rnd':
        rnd_demo()
        logger.info("Finish RND Demo")
    else:
        snd_demo()
        logger.info("Finish SND Demo")


