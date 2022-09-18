import scipy.stats as stats

class hypoTest():
	def wilcoxon(self, model1="BERTimbau", model2="word2vec", dataset="book reviews"):
		pass






medias = {
	'Geral': {
		'RVDC': [0.814, 0.762, 0.724, 0.704, 0.730, 0.821, 0.804, 0.801, 0.786, 0.810, 0.928, 0.822, 0.814, 0.748, 0.772, 0.818, 0.818, 0.818, 0.823, 0.840, 0.884, 0.822, 0.814, 0.748, 0.772, 0.818, 0.818, 0.818, 0.823, 0.840],
		'RVDE': [0.803, 0.724, 0.724, 0.716, 0.703, 0.809, 0.767, 0.770, 0.763, 0.782, 0.837, 0.003, 0.000, 0.800, 0.597, 0.786, 0.749, 0.749, 0.737, 0.763, 0.828, 0.814, 0.814, 0.751, 0.197, 0.281, 0.818, 0.822, 0.797, 0.823]
	}
}
testHip = dict()
for key in medias.keys():
	testHip[key] = stats.wilcoxon(medias[key]['RVDC'], medias[key]['RVDE'])
print(testHip)
