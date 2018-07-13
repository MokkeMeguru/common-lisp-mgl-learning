(defpackage fnn-learning
  (:use :mgl :cl)
  (:nicknames :f-learning))

(in-package :f-learning)

;; 入力には10個の整数を用います。
(defparameter *n-inputs* 10)
;; これから作られるFNNに学習させたい規則は入力される数 D に対して　(1 + D) mod  3 です。
;; つまり、 1 => 1 + 1 mod 3 = 2 / 100 => 1 + 100 mod 3 => 2 です。

(defparameter *n-outputs* 3)

;; 入力を読み込むための set-input 関数をあとで追加できるように
;; feed-forward ネットワークを定義します。
(defclass digit-fnn (fnn)
  ())

;; digit-fnn を ReLu を使った一つの隠れ層と、softmax 関数を用いた出力を用いて定義します。
;; * figure.
;;    [layer]     input   =>  hidden   =>    output
;;    [function]               relu          softmax
;;    [size]       i     i=>h         h=>o
;;
;;      i = *n-inputs*
;;      h = *n-hiddens*
;;      o = *n-outputs*

(defun make-digit-fnn (&key (n-hiddens 5))
  (build-fnn (:class 'digit-fnn)
    (input (->input :size *n-inputs*))
    (hidden-activation (->activation input :size n-hiddens))
    (hidden (->relu hidden-activation))
    (output-activation (->activation hidden :size *n-outputs*))
    (output (->softmax-xe-loss output-activation))))

;; このメソッドはジェネリック関数であり、つまり fnn のクラスである digit-fnn に紐付けられている関数です。
;; この関数はネットワークの入力層に入る前に処理されます。
;; note: 入力される数に対して one-hot ベクトルを作成しています。
;;       例: 2 => [0 1 0 ... 0]
(defmethod set-input (digits (fnn digit-fnn))
  (let* ((input (nodes (find-clump 'input fnn)))
         (output-lump (find-clump 'output fnn)))
    (fill! 0 input)
    (loop for i upfrom 0
          for digit in digits
          do (setf (mref input i digit) 1))
    (setf (target output-lump)
          (mapcar (lambda (digit)
                    (mod (1+ digit) *n-outputs*))
                  digits))))

;; note: mref と setf の使い方
;; * mref ... args: matrix column row
;; * setf ... args: element value
(let ((m (make-mat '(2 3) :initial-element 0.0)))
  (setf (mref m 1 2) 1.0)
  m)
;; #<MAT 2x3 B #2A((0.0d0 0.0d0 0.0d0) (0.0d0 0.0d0 1.0d0))>

;; 学習には、確率勾配降下によるクロスエントロピーを用いた損失関数を用いています。
(defun train-digit-fnn ()
  (let ((optimizer
          ;; ここで用いられている sgd は（日本では）Momentum SGD と呼ばれているものです。
          (make-instance 'segmented-gd-optimizer
                         :segmenter
                         (constantly
                          (make-instance 'sgd-optimizer
                                         :learning-rate 1
                                         :momentum 0.9
                                         :batch-size 100))))
        (fnn (make-digit-fnn)))
    ;; 並列して実行する fnn のインスタンス数を指定します。
    ;; 一般にはバッヂサイズかその約数が用いられます。
    (setf (max-n-stripes fnn) 50)
    ;; 重みの初期化を行います。
    (map-segments (lambda (weights)
                    ;; stddev : 標準偏差
                    (gaussian-random! (nodes weights) :stddev 0.01))
                  fnn)
    ;; 学習とテストの誤差をログとして記録するようにします。
    (monitor-optimization-periodically
     optimizer '((:fn log-test-error :period 10000)
                 (:fn reset-optimization-monitors :period 1000)))
    ;; 最後に最適化を行います。
    (minimize optimizer
              ;; fnn を bp-learner クラスでラップします。(monitors スロットを使うため)
              ;; monotors によって バッヂサイズ(100)ごとにログが書き足されます。
              (make-instance 'bp-learner
                             :bpn fnn
                             :monitors (make-cost-monitors
                                        fnn :attributes `(:event "train ")))
              ;; サンプル数は 10000 で学習を打ち止めします。
              :dataset (make-sampler 10000))))

;; max-n-samples 分のサンプルを吐く sampler オブジェクト を作成します。
(defun make-sampler (max-n-samples)
  (make-instance 'function-sampler :max-n-samples max-n-samples
                 :generator (lambda () (random *n-inputs*))))

;; テストの誤差をログに記録する他、学習ののはじめの最適化の状況や bpn について記録します。
;; 記録部は一定の学習周期ごとに呼び出されています。
(defun log-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (describe optimizer)
    (describe (bpn learner)))
  (log-padded
   (monitor-bpn-results (make-sampler 1000) (bpn learner)
                        (make-cost-monitors
                         (bpn learner) :attributes `(:events "pred.")))))

;; (repeatably ()
;;   (let ((*log-time* nil))
;;     (train-digit-fnn)))

;; raining at n-instances: 0
;; train  cost: 0.000e+0 (0)
;; #<SEGMENTED-GD-OPTIMIZER {100B2E8BD3}>
;; SEGMENTED-GD-OPTIMIZER description:
;;   N-INSTANCES = 0
;;   OPTIMIZERS = (#<SGD-OPTIMIZER
;;                   #<SEGMENT-SET
;;                     (#<->WEIGHT (HIDDEN OUTPUT-ACTIVATION) :SIZE 15 1/1 :NORM 0.04473>
;;                      #<->WEIGHT (:BIAS  OUTPUT-ACTIVATION) :SIZE  3 1/1 :NORM 0.01850>
;;                      #<->WEIGHT (INPUT  HIDDEN-ACTIVATION) :SIZE 50 1/1 :NORM 0.07159>
;;                      #<->WEIGHT (:BIAS  HIDDEN-ACTIVATION) :SIZE  5 1/1 :NORM 0.03056>)
;;                      {100B2F4D03}> {100B2E8B03}>)
;;   SEGMENTS = (#<->WEIGHT (HIDDEN OUTPUT-ACTIVATION) :SIZE 15 1/1 :NORM 0.04473>
;;               #<->WEIGHT (:BIAS OUTPUT-ACTIVATION)  :SIZE   3 1/1 :NORM 0.01850>
;;               #<->WEIGHT (INPUT HIDDEN-ACTIVATION)  :SIZE  50 1/1 :NORM 0.07159>
;;               #<->WEIGHT (:BIAS HIDDEN-ACTIVATION)  :SIZE   5 1/1 :NORM 0.03056>)

;; #<SGD-OPTIMIZER {100B2E8B03}>
;; GD-OPTIMIZER description:
;;   N-INSTANCES = 0
;;   SEGMENT-SET = #<SEGMENT-SET
;;                     (#<->WEIGHT (HIDDEN OUTPUT-ACTIVATION) :SIZE 15 1/1 :NORM 0.04473>
;;                      #<->WEIGHT (:BIAS OUTPUT-ACTIVATION)  :SIZE 3 1/1 :NORM 0.01850>
;;                      #<->WEIGHT (INPUT HIDDEN-ACTIVATION)  :SIZE 50 1/1 :NORM 0.07159>
;;                      #<->WEIGHT (:BIAS HIDDEN-ACTIVATION)  :SIZE  5 1/1 :NORM 0.03056>)
;;                      {100B2F4D03}>
;;   LEARNING-RATE = 1.00000e+0
;;   MOMENTUM = 9.00000e-1
;;   MOMENTUM-TYPE = :NORMAL
;;   WEIGHT-DECAY = 0.00000e+0
;;   WEIGHT-PENALTY = 0.00000e+0
;;   N-AFTER-UPATE-HOOK = 0
;;   BATCH-SIZE = 100

;; BATCH-GD-OPTIMIZER description:
;;   N-BEFORE-UPATE-HOOK = 0
;; #<DIGIT-FNN {100B2E8CB3}>
;; BPN description:
;;   CLUMPS = #(#<->INPUT INPUT :SIZE 10 1/50 :NORM 0.00000>
;;              #<->ACTIVATION (HIDDEN-ACTIVATION :ACTIVATION) :STRIPES 1/50 :CLUMPS 4>
;;              #<->RELU HIDDEN :SIZE 5 1/50 :NORM 0.00000>
;;              #<->ACTIVATION (OUTPUT-ACTIVATION :ACTIVATION) :STRIPES 1/50 :CLUMPS 4>
;;              #<->SOFTMAX-XE-LOSS OUTPUT :SIZE 3 1/50 :NORM 0.00000>)
;;   N-STRIPES = 1
;;   MAX-N-STRIPES = 50
;; pred. cost: 1.100d+0 (1000.00)
;; training at n-instances: 1000
;; train  cost: 1.093d+0 (1000.00)
;; training at n-instances: 2000
;; train  cost: 5.886d-1 (1000.00)
;; training at n-instances: 3000
;; train  cost: 3.574d-3 (1000.00)
;; training at n-instances: 4000
;; train  cost: 1.601d-7 (1000.00)
;; training at n-instances: 5000
;; train  cost: 1.973d-9 (1000.00)
;; training at n-instances: 6000
;; train  cost: 4.882d-10 (1000.00)
;; training at n-instances: 7000
;; train  cost: 2.771d-10 (1000.00)
;; training at n-instances: 8000
;; train  cost: 2.283d-10 (1000.00)
;; training at n-instances: 9000
;; train  cost: 2.123d-10 (1000.00)
;; training at n-instances: 10000
;; train  cost: 2.263d-10 (1000.00)
;; pred. cost: 2.210d-10 (1000.00)
;; (#<->WEIGHT (:BIAS HIDDEN-ACTIVATION) :SIZE 5 1/1 :NORM 2.94294>
;;  #<->WEIGHT (INPUT HIDDEN-ACTIVATION) :SIZE 50 1/1 :NORM 11.48995>
;;  #<->WEIGHT (:BIAS OUTPUT-ACTIVATION) :SIZE 3 1/1 :NORM 3.39103>
;;  #<->WEIGHT (HIDDEN OUTPUT-ACTIVATION) :SIZE 15 1/1 :NORM 11.39339>)

(defparameter
    results
  (repeatably ()
    (let ((*log-time* nil))
      (train-digit-fnn))))
;; results
;; (#<->WEIGHT (:BIAS HIDDEN-ACTIVATION) :SIZE 5 1/1 :NORM 2.94294>
;;             #<->WEIGHT (INPUT HIDDEN-ACTIVATION) :SIZE 50 1/1 :NORM 11.48995>
;;             #<->WEIGHT (:BIAS OUTPUT-ACTIVATION) :SIZE 3 1/1 :NORM 3.39103>
;;             #<->WEIGHT (HIDDEN OUTPUT-ACTIVATION) :SIZE 15 1/1 :NORM 11.39339>)


;; note : cuda を使うと精度が怪しくなる。
;; (with-cuda* ()
;;     (repeatably ()
;;       (let ((*log-time* nil))
;;         (train-digit-fnn))))
;;  => ... pred. cost: 5.382d-9 (1000.00)

;; note
;; CLUMP は　LUMP または　BPN　を示しており、微分可能な関数です。
;; BPN は FNN や RNN の抽象クラスです。

