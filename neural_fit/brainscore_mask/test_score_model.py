from candidate_models import score_model
from candidate_models.model_commitments import brain_translated_pool

identifier = 'resnet-18'
model = brain_translated_pool[identifier]
#score = score_model(model_identifier=identifier, model=model, benchmark_identifier='dicarlo.Majaj2015.IT-pls')
score = score_model(model_identifier=identifier, model=model, benchmark_identifier='dicarlo.Majaj2015.V4-pls')
print(score)
