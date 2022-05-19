import pandas as pd
import sys

def read_file(fname,phenomena=None):
	"""
	Basic file reading helper function
	returns source, correct translation, bad translation,reference
	"""
	data = pd.read_csv(fname,separator='\t')
	if phenomena is not None:
		data = data[data['phenomena']==phenomena]
	return data['source'], data['good-translation'], data['incorrect-translation'], data['reference']

def get_sentence_bleu_scores(source, good_translation, incorrect_translation, reference, use_reference=True):
	"""
	Function to compute sentence-bleu
	returns good_translation_scores, incorrect_translation_scores
	"""
	from sacrebleu import sentence_bleu

	good_translation_scores = [sentence_bleu(hyp,[ref]).score for hyp, ref in zip(good_translation, reference)]
	incorrect_translation_scores = [sentence_bleu(hyp,[ref]).score for hyp, ref in zip(incorrect_translation, reference)]

	return good_translation_scores, incorrect_translation_scores

def get_sentence_chrf_scores(source, good_translation, incorrect_translation, reference, use_reference=True):
	"""
	Function to compute sentence-bleu
	returns good_translation_scores, incorrect_translation_scores
	"""
	from sacrebleu import sentence_chrf

	good_translation_scores = [sentence_chrf(hyp,[ref]).score for hyp, ref in zip(good_translation, reference)]
	incorrect_translation_scores = [sentence_chrf(hyp,[ref]).score for hyp, ref in zip(incorrect_translation, reference)]

	return good_translation_scores, incorrect_translation_scores

def get_comet_scores(source, good_translation, incorrect_translation, reference, use_reference=True,model_path='wmt20-comet-da'):
	"""
	Function to compute comet-score
	returns good_translation_scores, incorrect_translation_scores
	"""
	from comet import download_model, load_from_checkpoint
	if not use_reference:
		assert 'qe' in model_path
	model_path = download_model(model_path)
	model = load_from_checkpoint(model_path)

	if not use_reference:
		reference = ["" for i in source]

	data = {"src": source, "mt": good_translation, "ref": reference}
	data = [dict(zip(data, t)) for t in zip(*data.values())]
	good_translation_scores, good_translation_sys_score = model.predict(data, gpus=1, batch_size=16)

	data = {"src": source, "mt": incorrect_translation, "ref": reference}
	data = [dict(zip(data, t)) for t in zip(*data.values())]
	good_translation_scores, good_translation_sys_score = model.predict(data, gpus=1, batch_size=16)


	return good_translation_scores, incorrect_translation_scores

def kendal_tau(good_translation_scores, incorrect_translation_scores):
	"""
	please check if the calculation is right
	"""
	concordant = np.sum(good_translation_scores>incorrect_translation_scores)
	discorndant = np.sum(good_translation_scores<=incorrect_translation_scores)

	tau = (concordant - discorndant) / (concordant + discorndant)

	return tau


