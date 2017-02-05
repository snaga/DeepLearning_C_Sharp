# DeepLearning_C_Sharp

書籍「[ゼロから作るDeep Learning――Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)」で解説されているPythonのプログラムをC#で再実装してみるプロジェクトです。

## 目的

* C#プログラミングスキルの獲得
* ニューラルネットワーク、ディープラーニングの知識の獲得

## 方針

* 書籍に記載されているPythonのアルゴリズムを極力踏襲して実装する
* numpyのndarrayはC#の二次元配列を使って実現する
* numpy関連の処理は、同等の関数群をC#で実装する
* Pythonに依存するデータ構造（Pickle化されたndarrayとか）は、Pythonを使って一旦CSVに落としてからC#で読み込んで使う。

