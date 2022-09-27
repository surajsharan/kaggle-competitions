import pandas as pd
import numpy as np
from sklearn import model_selection


def create_folds(data, num_splits):
    data["kfold"] = -1
    kf = model_selection.StratifiedKFold(
        n_splits=num_splits, shuffle=True, random_state=2021)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['language'])):
        data.loc[v_, 'kfold'] = f
    return data


def convert_answers(row):
    return {'answer_start': [row[0]], 'text': [row[1]]}


if __name__ == "__main__":

    train = pd.read_csv('input/train.csv')
    external_mlqa = pd.read_csv('External/mlqa_hindi.csv')
    external_xquad = pd.read_csv('External/xquad.csv')
    external_train = pd.concat([external_mlqa, external_xquad])

    train = create_folds(train, num_splits=5)

    external_train["kfold"] = -1
    external_train['id'] = list(np.arange(1, len(external_train)+1))
    train = pd.concat([train, external_train]).reset_index(drop=True)

    train['answers'] = train[['answer_start', 'answer_text']].apply(
        convert_answers, axis=1)
    train.to_csv('input/train_folds.csv', index=False)
