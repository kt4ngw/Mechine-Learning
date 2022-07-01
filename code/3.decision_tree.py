# 准备的库 1 数据划分库 2 决策树分类库 3 鸢尾花数据集生成库 4 忽略警告库
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
import warnings
warnings.filterwarnings("ignore")

# 生成鸢尾花数据集
X, y = load_iris(return_X_y=True)
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# 训练模型
Dec_clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
# ------------ #
# 评估模型 AUC
# 测试集上评估模型的准确率AUC为100%
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, Dec_clf.predict(X_test)))
# ------------ #
feature_name = load_iris().feature_names
import graphviz
dot_data = tree.export_graphviz(Dec_clf, feature_names=feature_name, class_names=['0', '1', '2'],
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)

dot_data.view()