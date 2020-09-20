# spam_classification

signateの練習課題（スパムメール分類）課題用。
https://signate.jp/competitions/104

## コードの内容

- Experiment.py : 本体。データの読み込みから学習まで実行する。
- util
   - Dataset.py : spamデータセットの作成
   - Transformaer.py : テキストの前処理
   - Trainer.py : モデル学習（Random Forest, Naive Bayes, SVM. グリッドサーチ交差検証）
   - Storage.py : 結果の保存
- config
   - param.json : ハイパーパラメータ等の設定ファイル。