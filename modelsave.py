import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import itertools

from lrclass import linear_regression

df = pd.read_csv('placement_with_rating.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
param_grid = {
    'l_rate': [0.001, 0.0001, 0.00001],       #
    'no_iter': [50, 100, 150]
}
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=50)
results = {}
for params in all_params:
    current_l_rate = params['l_rate']
    current_no_iter = params['no_iter']
    param_key = f"lr={current_l_rate}_iter={current_no_iter}" 
    print(f"\nTesting combination: {param_key}")

    fold_scores = [] 

    for fold_num, (train_index, test_index) in enumerate(kf.split(X)):
       

        
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        Y_train_fold, Y_test_fold = Y[train_index], Y[test_index]

      
        model_fold = regression(l_rate=current_l_rate, no_iter=current_no_iter)

 
        model_fold.fit(X_train_fold, Y_train_fold)

        try:
            test_pred_fold = model_fold.predict(X_test_fold)
        
            r2 = r2_score(Y_test_fold, test_pred_fold)
            fold_scores.append(r2)
        except Exception as e:
            print(f"  Error during prediction/scoring in Fold {fold_num + 1}: {e}")
            fold_scores.append(-np.inf) 

    average_r2_for_combo = np.mean(fold_scores)
    results[param_key] = average_r2_for_combo
    print(f"  Average R2 across {K} folds: {average_r2_for_combo:.4f}")


best_param_key = max(results, key=results.get)
best_average_r2 = results[best_param_key]

best_lr = float(best_param_key.split('_')[0].split('=')[1])
best_iter = int(best_param_key.split('_')[1].split('=')[1])



final_model = regression(l_rate=best_lr, no_iter=best_iter)
final_model.fit(X, Y) 
with open('model.pkl', 'rb') as f:
    classifier = pickle.load(f)
