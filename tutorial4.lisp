(defpackage mgl-learning
  (:use :mgl :cl
        :mgl-dataset
        :mgl-resample)
  (:nicknames :m-learning))

(in-package :m-learning)

;;; 見方 --------------------------------------------------------------------
;; ;;; タイトル -------------------------...
;; ;; functions
;; ;; * 関数名
;; ;;    args : 引数
;; ;;    return : 返り値
;; ;;
;; ;; variable
;; ;; * 変数名
;; ;;    説明
;; ;;
;; ;; class
;; ;; * クラス名
;; ;;    フィールド名 : 型 : 説明
;;
;;   説明コード
;;
;; ;;; --------------------------------...
;;; ------------------------------------------------------------------------


;;; Datasets ----------------------------------------------------------------
;; functions
;; * map-dataset
;;    args : lambda list
;;    return : list
;; * map-datasets
;;    args : lambda list's-list
;;    return : list's-list

;; 単一のリストからデータセットを作るには以下のようにします。
(map-dataset #'prin1 '(1 2 3 (4 5))) ;; 123(4 5)

;; 複数のリストを連結し、データセットを作るには以下のようにします。
(map-datasets #'prin1 '((0 1 2) (:a :b))) ;; (0 :A) (1 :B)

;; :impute は欠損している値を埋める場合に、その値を指定するために用いられます。
(map-datasets #'prin1 '((0 1 2) (:a :b)) :impute nil) ;; (0 :A)(1 :B)(2 NIL)
(map-datasets #'prin1 '((0 1 2) (:a :b)) :impute :c) ;; (0 :A)(1 :B)(2 :C)
(map-datasets #'prin1 '((0 1 2) (3 4 5) (6 7 8))) ;; (0 3 6)(1 4 7)(2 5 8)

;; 以下のようにサンプラー(後述)を用いたシーケンスを用いて作成することもできます。
(map-datasets #'prin1
              (list '(0 1 2)
                    (make-sequence-sampler
                     '(:a :b) :max-n-samples 2))) ;; (0 :A)(1 :B)
(map-datasets #'prin1
              (list '(0 1 2)
                    (make-sequence-sampler
                     '(:a :b :c) :max-n-samples 2))) ;; (0 :A)(1 :B)
(map-datasets #'prin1
              (list '(0 1 2)
                    (make-sequence-sampler
                     '(:a :b :c) :max-n-samples 3))) ;; (0 :A)(1 :B)(2 :C)
;;; --------------------------------------------------------------------------

;;; Samplers -----------------------------------------------------------------
;; functions
;; * sample [generic]
;;    args : sampler
;;    return : function
;; * finishedp [generic]
;;    args : sampler
;;    return : bool
;; * list-samples
;;    args : sampler max-size
;;    return : list-of-samples
;; * make-sequence-sampler
;;    args : seq &key max-n-samples
;;    return : sampler
;; * make-random-sampler
;;    args : seq &key max-n-samples
;;    return : sampler
;; variable
;; * *infinity-empty-dataset*
;;    無限長の nil のリスト
;; class
;; * function-sampler
;;    :generator : function : サンプルの生成を行う関数
;;    :max-n-samples : num : 最大サンプル数 (デフォルト値 nil)
;;    :name : string : このインスタンスの名前 (print する際に用いられる) (デフォルト値 nil)
;;    :n-samples : num : サンプル数 (デフォルト値 0)

;; *infinity-empty-dataset* の使い方
(list-samples *infinitely-empty-dataset* 10) ;; (NIL NIL NIL NIL NIL NIL NIL NIL NIL NIL)

;; function-sampler のインスタンスの使い方
;; max-n-sample 個だけサンプルを作ります。
(list-samples (make-instance 'function-sampler
                             :generator (lambda () (random 10))
                             :max-n-samples 2)
              10) ;; (7 9)

;; list-samples で インスタンスが 12 個作ったサンプルの内の 10 個を取り出しています。
(list-samples (make-instance 'function-sampler
                             :generator (lambda () (random 10))
                             :max-n-samples 12)
              10) ;; (2 1 8 8 5 3 1 2 4 5)

;; おそらく max-n-samples - n-samples の個数だけサンプルが返ってきます。 [要確認]
(list-samples (make-instance 'function-sampler
                             :generator (lambda () (random 10))
                             :max-n-samples 10
                             :n-samples 7)
              10) ;; (1 3 6)

;; :name スロットに名前を入れることで print した際に名前が表示されます。
(defparameter sampler (make-instance 'function-sampler
                                     :generator (lambda () (random 10))
                                     :name "SAMPLER"))

sampler ;; #<FUNCTION-SAMPLER "SAMPLER" >

;; max-n-samples を指定していない場合、要求されたサンプル数に対して、取れるだけのサンプルを渡します。
(list-samples (make-instance 'function-sampler
                             :generator (lambda () (random 10))
                             :name "SAMPLER")
              10) ;; (9 1 4 3 5 0 9 7 2 5)

;;; --------------------------------------------------------------------------------------

;;; Resampling ---------------------------------------------------------------------------
;;; Partitions
;; function
;; * fracture
;;    args : fractions seq &key weight
;;    return : list's-list
;; * stratify
;;    args : seq &key (key #'identity) (test #'eql)
;;    return : list's-list
;; * fracture-stratified
;;    args : fractions seq &key (key #'identity) (test #'eql) weight
;;    return : list's-list

;; fractions には num の他、分割する方法を示したリストを与えることができます。
(fracture 3 '(0 1 2 3 4 5)) ;; ((0 1) (2 3) (4 5))
(fracture '(1 2 3) '(0 1 2 3 4 5)) ;; ((0) (1 2) (3 4 5))

;; 重み weigt による分割の変化
(fracture 3 '(0 1 2 3 4) :weight (lambda (a) (if (eq a 4) 0 1))) ;; ((0) (1 2) (3 4))
(fracture 3 '(0 1 2 3 4) :weight (lambda (a) (if (eq a 0) 0 1))) ;; ((0 1) (2 3) (4))

;; stratify では key でリストの各要素にキーを与え、 test でそれをクラスタリングします。
;; 例えば、key で偶数か奇数かを分類して test で結果が等しいものごとに分類すると以下のようになります。
(stratify '(0 1 2 3 4 5) :key #'evenp :test 'eql) ;; ((0 2 4) (1 3 5))

;; key が #'identity、 test が #'eq のときは省略できます。
(stratify '(0 1 2 3 4 5)) ;; ((0) (1) (2) (3) (4) (5))

;; fracture-stratify では stratify で分割したリストの要素を同じ比率だけ含んだリスト群を作成します。
(fracture-stratified 2 '(0 1 2 3 4 5 6 7 8 9) :key #'evenp) ;; ((0 2 1 3) (4 6 8 5 7 9))
;;;

;;; Cross-validation
;; function
;; * cross-validation
;;    args : data fn &key (n-folds 5) (folds (alexandria.0.DEV:IOTA N-FOLDS))
;;          (SPLIT-FN #'SPLIT-FOLD/MOD) PASS-FOLD
;; * split-fold/mod                     [要学習]
;;    args : seq fold n-folds
;; * split-fold/cont                    [要学習]
;;    args : seq fold n-folds
;; * split-stratified                   [要学習]
;;    args : seq fold n-folds &key (key #'identity) (test #'eql) weight
(cross-validate '(0 1 2 3 4)
                (lambda (test training)
                  (list test training)
                  :n-folds 5))
;; (((0) (1 2 3 4))
;;  ((1) (0 2 3 4))
;;  ...
;;  ((4) (0 1 2 3)))

(cross-validate '(0 1 2 3 4)
                (lambda (fold test training)
                  (list :fold fold test training))
                :folds '(2 3)
                :pass-fold t)
;; ((:FOLD 2 (2) (0 1 3 4))
;;  (:FOLD 3 (3) (0 1 2 4)))

(cross-validate '(0 1 2 3 4)
                (lambda (fold test training)
                  (list :fold fold test training))
                :folds '(2 3 4)
                :pass-fold t)
;; ((:FOLD 2 (2) (0 1 3 4)) (:FOLD 3 (3) (0 1 2 4)) (:FOLD 4 (4) (0 1 2 3)))
;;;
;;; Bagging
;; function
;; * bag                               [要学習]
;;    args : seq fn &key (ratio 1) n weight (replacement t) key
;;           (test #'eql) (random-state *random-state*)
;; * sample-from
;;    args : ratio seq &key weight replacement (random-state *random-state*)
;; * sample-stratified                 [要学習]
;;    args : ratio seq &key weight replacement (key #'identity) (test #'eql)
;;           (random-state *random-state*)

;; データから ratio 分だけサンプルを抽出します。
(sample-from 1/2 '(0 1 2 3 4 5)) ;; (2 5 4)

(sample-from 1 '(0 1 2 3 4 5)) ;; (0 3 4 2 1 5)

;; replacement を t にすると重複を許します。
(sample-from 1 '(0 1 2 3 4 5) :replacement t) ;; (5 4 1 1 2)

(sample-from 1/2 '(0 1 2 3 4 5 6 7 8 9) :weight #'identity) ;; (2 3 7 1 4 5 0 8)
;; sum -> 30 = 45/2 : 45 is (+ 0 1 2 ... 9)
;;;
;;; CV-Bagging
;; function
;; * bag-cv
;;    args : data fn &key n (n-folds 5) (folds (alexandra.0.dev:IOTA n-folds))
;;           (split-fn #'split-fold/mod) pass-fold (random-state *radom-state*)

;; ---------------------------------------------------------------------------------------
