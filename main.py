from preprocessing.main_preprocessing import preprocess
from models.baseline import create_mean_reg, create_KNN_mean_reg, create_linear_reg, create_random_predictor
from models.boosting import create_ada_boost, create_gradient_boosted_tree, create_xg_boost
from utils import print_eval


def main():
    print('Preprocessing...\n')
    ###############    Parameters should be modified here    ###############
    X_train, X_test, Y_train, Y_test, columns = preprocess(
        path_to_folder='data',
        remove_outliers=True,
        imputing_missing_values=False,
        rescaling=False,
        pca=False,
        pls=False,
    )

    ###############    Model should be selected here    ###############
    # model = create_mean_reg()
    # model = create_KNN_mean_reg(columns)
    model = create_linear_reg()
    # model = create_random_predictor()
    # model = create_ada_boost()
    # model = create_gradient_boosted_tree()
    # model = create_xg_boost()

    print('Training model...\n')
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    print_eval(Y_test, predictions)


if __name__ == '__main__':
    main()
