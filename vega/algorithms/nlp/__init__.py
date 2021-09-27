from vega.common.class_factory import ClassFactory


ClassFactory.lazy_register("vega.algorithms.nlp", {
    "bert_trainer_callback": ["BertTrainerCallback"],
    "src.bert_for_pre_training": ["BertNetworkWithLoss"],
})
