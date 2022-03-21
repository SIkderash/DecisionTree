from random import seed
from random import randrange
from csv import reader


def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset


def str_column_to_float(dataset):
	for column in range(len(dataset[0])):
		for row in dataset:
			row[column] = float(row[column].strip())


def cross_validation_split(dataset, n_folds):
	dataset_split = []
	dataset_copy = dataset
	fold_size = len(dataset) // n_folds
	for i in range(n_folds):
		fold = []
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = []
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = []
		actual = []
		for row in fold:
			row_copy = list(row)
			row_copy[-1] = None
			test_set.append(row_copy)
			actual.append(row[-1])
		predicted = algorithm(train_set, test_set, *args)
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores


def split_by_attribute_value(index, value, dataset):
	left, right = [], []
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right


def get_number_of_instances(groups):
	instances = []
	for group in groups:
		instances.append(len(group))

	n_instances = sum(instances)
	n_instances = float(n_instances)
	return n_instances


def gini_index(groups, classes):
	total_instances = get_number_of_instances(groups)
	gini = 0.0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0.0
		all_labels = []
		for row in group:
			all_labels.append(row[-1])
		for class_val in classes:
			p = all_labels.count(class_val) / size
			score += p * p
		gini += (1.0 - score) * (size / total_instances)
	return gini


def get_unique_class_values(dataset):
	class_values_list = []
	for row in dataset:
		class_values_list.append(row[-1])

	return set(class_values_list)


def get_best_split(dataset):
	class_values = get_unique_class_values(dataset)
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	number_of_attributes = len(dataset[0])-1

	for index in range(number_of_attributes):
		for row in dataset:
			attribute_value = row[index]
			groups = split_by_attribute_value(index, attribute_value, dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, attribute_value, gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}


def get_label(group):
	outcomes = [row[-1] for row in group]
	outcomes_set = set(outcomes)
	mx = 0
	label = None
	for outcome in outcomes_set:
		if outcomes.count(outcome) > mx:
			mx = outcomes.count(outcome)
			label = outcome
	return label


def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left'] = node['right'] = get_label(left + right)
		return
	if depth >= max_depth:
		node['left'], node['right'] = get_label(left), get_label(right)
		return
	if len(left) <= min_size:
		node['left'] = get_label(left)
	else:
		node['left'] = get_best_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	if len(right) <= min_size:
		node['right'] = get_label(right)
	else:
		node['right'] = get_best_split(right)
		split(node['right'], max_depth, min_size, depth+1)


def build_tree(train, max_depth, min_size):
	root = get_best_split(train)
	split(root, max_depth, min_size, 1)
	return root


def is_internal(node):
	return isinstance(node, dict)


def predict(node, row):
	if row[node['index']] < node['value']:
		if is_internal(node['left']):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if is_internal(node['right']):
			return predict(node['right'], row)
		else:
			return node['right']


def decision_tree(train, test, max_depth, min_size):
	root = build_tree(train, max_depth, min_size)
	printTree(root, 0)
	predictions = []
	for row in test:
		prediction = predict(root, row)
		predictions.append(prediction)

	return predictions


def printTree(node, space):
	print(space * "    " , end = " ")
	if is_internal(node):
		print("Attribute ",node['index'], "<", node['value'])
		printTree(node['left'], space+1)
		printTree(node['right'], space+1)
	else:
		print(node)
		return


seed(1)
n_folds = 10
max_depth = 5
min_size = 5

filename = 'wine.csv'
dataset = load_csv(filename)
str_column_to_float(dataset)

scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: ', scores)
print('Mean Accuracy: ', (sum(scores)/float(len(scores))))
