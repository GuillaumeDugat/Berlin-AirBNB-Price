from preprocessing.main_preprocessing import preprocess
from models.baseline import create_mean_reg, create_KNN_mean_reg, create_linear_reg, create_random_predictor
from models.boosting import create_ada_boost, create_gradient_boosted_tree, create_xg_boost
from models.bagging import create_bagging_model
from models.svm import create_SVR_model
from models.trees import create_decision_tree_model, create_random_forest_model
from utils import print_eval


def main():
    print('Preprocessing...\n')
    ###############    Parameters should be modified here    ###############
    X_train, X_test, Y_train, Y_test, columns = preprocess(
        path_to_folder='data',
        remove_outliers=True,
        imputing_missing_values=False,
        rescaling=True,
        pca=False,
        pls=False,
        forward_selection=False,
        backward_selection=False,
    )

    ###############    Model should be selected here    ###############
    # model = create_mean_reg()
    # model = create_KNN_mean_reg(columns)
    model = create_linear_reg()
    # model = create_random_predictor()
    # model = create_ada_boost()
    # model = create_gradient_boosted_tree()
    # model = create_xg_boost()
    # model = create_bagging_model()
    # model = create_SVR_model()
    # model = create_decision_tree_model()
    # model = create_random_forest_model()

    print('Training model...\n')
    model.fit(X_train, Y_train)

    if "best_params_" in dir(model):
        print('Best parameters:', model.best_params_)

    predictions = model.predict(X_test)
    print_eval(Y_test, predictions)


if __name__ == '__main__':
    main()
