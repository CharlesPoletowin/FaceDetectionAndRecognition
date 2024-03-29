from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
# print(faces.target_names)
# print(faces.images.shape)

#画一些人脸，看看需要处理的数据
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
# fig, ax = plt.subplots(3,5)
# fig.subplots_adjust(left=0.0625, right=1.2, wspace=1)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i], cmap='bone')
#     axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])

# plt.show()


#使用预处理来提取更有意义的特征。这里使用主成份分析来提取150个基本元素，然后将其提供给支持向量机分类器。
#将这个预处理和分类器打包成管道

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

#为了测试分类器的训练效果，将数据集分解成训练集和测试集进行交叉检验
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=42)

#用网络搜索交叉检验来寻找最优参数组合。通过不断调整C（松弛变量）和参数gamma（控制径向基函数核的大小），确定最优模型
from sklearn.grid_search import GridSearchCV
param_grid = {'svc__C': [1,5,10,50], 'svc__gamma':[0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)

grid.fit(x_train, y_train)
print(grid.best_params_)

model = grid.best_estimator_
y_fit = model.predict(x_test)
# #比较预测结果和真实结果
# fig, ax = plt.subplots(4, 6)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(x_test[i].reshape(62, 47), cmap='bone')
#     axi.set(xticks=[], yticks=[])
#     axi.set_ylabel(faces.target_names[y_fit[i]].split()[-1],
#                   color='black' if y_fit[i] == y_test[i] else 'red')
# fig.suptitle('Predicted Names; Incorect Lables in Red', size=14)
# plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_fit, target_names=faces.target_names))

#画出混淆矩阵，它可以帮助我们清晰的判断那些标签容易被分类器误判
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_fit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
