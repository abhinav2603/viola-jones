import numpy as np
import math
def integral_image(image):
	s = np.zeros(image.shape)
	ii = np.zeros(image.shape)
	for y in range(len(image)):
		for x in range(len(image[y])):
			if y == 0:
				s[y][x] = image[y][x]
			else:
				s[y][x] = s[y-1][x] + image[y][x]
			if x == 0:
				ii[y][x] = s[y][x]
			else:
				ii[y][x] = ii[y][x-1] + s[y][x]
	return ii

class RectangleRegion:
	def __init__(self, x, y, width, height):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
	def compute_feature(self, ii):
		return ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] - (ii[self.y+self.height][self.x]+ii[self.y][self.x+self.width])

class WeakClassifier:
	def __init__(self, positive_regions, negative_regions, threshold, polarity):
		self.positive_regions = positive_regions
		self.negative_regions = negative_regions
		self.threshold = threshold
		self.polarity = polarity
	def classify(self, x):
		feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions]) - sum([neg.compute_feature(ii) for neg in self.negative_regions])
		return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0

class ViolaJones:
	def __init__(self, T = 10):
		self.T = T
		self.alphas = []	
		self.clfs = []

	def build_features(self, image_shape):
		height, width = image_shape
		features = []
		for w in range(1, width+1):
			for h in range(1, height+1):
				i = 0
				while i + w < width:
					j = 0
					while j + h < height:
						#2 rectangle features
						immediate = RectangleRegion(i, j, w, h)
						right = RectangleRegion(i+w, j, w, h)
						if i + 2 * w < width: #Horizontally Adjacent
							features.append(([right], [immediate]))
						bottom = RectangleRegion(i, j+h, w, h)
						if j + 2 * h < height: #Vertically Adjacent
							features.append(([immediate], [bottom]))
						right_2 = RectangleRegion(i+2*w, j, w, h)
						#3 rectangle features
						if i + 3 * w < width: #Horizontally Adjacent
							features.append(([right], [right_2, immediate]))
						bottom_2 = RectangleRegion(i, j+2*h, w, h)
						if j + 3 * h < height: #Vertically Adjacent
							features.append(([bottom], [bottom_2, immediate]))
						#4 rectangle features
						bottom_right = RectangleRegion(i+w, j+h, w, h)
						if i + 2 * w < width and j + 2 * h < height:
							features.append(([right, bottom], [immediate, bottom_right]))
					j += 1
				i += 1
		return features

	def apply_features(self, features, training_data):
		X = np.zeros((len(features),len(training_data)))
		y = np.array(map(lambda data: data[1], training_data))
		i = 0
		for positive_regions, negative_regions in features:
			feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - sum([neg.compute_feature(ii) for neg in negative_regions])
			X[i] = list(map(lambda data: feature(data[0]), training_data))
			i += 1
		return X, y

	def train(self,training,pos_num,neg_num):
		training_data = []
		weights = np.zeros(len(training))
		for x in range(len(training)):
			training_data.append((integral_image(training[x][0]), training[x][1]))
		if training[x][1] == 1:
			weights[x] = 1.0 / (2 * pos_num)
		else:
				weights[x] = 1.0 / (2 * neg_num)
		features = self.build_features(training_data[0][0].shape)
		X, y = self.apply_features(features, training_data)
		for t in range(self.T):
			weights = weights / np.linalg.norm(weights)
			weak_classifiers = self.train_weak(X, y, features, weights)
			clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
			beta = error / (1.0 - error)
			for i in range(len(accuracy)):
				weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
			self.alphas.append(math.log(1.0/beta))
			self.clfs.append(clf)

	def train_weak(X, y, features, weights):
		t_plus, t_minus = 0,0
		for w,label in zip(weights,y):
			if y == 1:
				t_plus += w
			else:
				t_minus += w

		classifiers = []
		total_features = X.shape[0]
		for index, feature in enumerate(X):
			if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
				print("Trained %d classifiers out of %d" % (len(classifiers), total_features))
			applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
			pos_seen, neg_seen = 0, 0
			pos_weights, neg_weights = 0, 0
			min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
			for w, f, label in applied_feature:
				error = min(neg_weights + t_plus - pos_weights, pos_weights + t_minus - neg_weights)
				if error < min_error:
					min_error = error
					best_feature = features[index]
					best_threshold = f
					best_polarity = 1 if pos_seen > neg_seen else -1
				if label == 1:
					pos_seen += 1
					pos_weights += w
				else:
					neg_seen += 1
					neg_weights += w
			clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
			classifiers.append(clf)
		return classifiers

	def select_best(self, classifiers, weights, training_data):
		best_clf, best_error, best_accuracy = None, float('inf'), None
		for clf in classifiers:
			error,accuracy = 0,[]
			for data,w in zip(training_data,weights):
				correctness = abs(clf.classify(data[0])-data[1])
				accuracy.append(correctness)
				error += w*correctness
			error /= len(training_data)
			if error < best_error:
				best_error, best_clf, best_accuracy = error, clf, accuracy
		return best_clf, best_error, best_accuracy

	def classify(self, image):
		total = 0
		ii = integral_image(image)
		for alpha, clf in zip(self.alphas, self.clfs):
			total += alpha * clf.classify(ii)
		return 1 if total >= 0.5 * sum(self.alphas) else 0


