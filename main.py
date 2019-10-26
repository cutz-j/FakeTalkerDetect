from train import alexnet_train, siamese_train
from evaluate import alexnet_evaluate, siamese_evaluate

alexnet = alexnet_train()
siamese = siamese_train()
alexnet_evaluate()
siamese_evaluate(siamese)